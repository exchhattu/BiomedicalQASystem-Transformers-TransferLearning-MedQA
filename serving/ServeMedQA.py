"""
Rojan Shresetha, Insight Data Science
Wed Jan 29 09:54:09 2020
"""


import os
import io
import json
import numpy as np
import collections

import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F

from torch.utils.data import (DataLoader, 
                                RandomSampler, 
                                SequentialSampler,
                                TensorDataset)

from pytorch_transformers import  XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer
from utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended,
                         get_final_text)

class MedQAInference:

    def __init__(self):
        self._path_to_checkpoint = None # checkpoint_file_path = None
        self._model = None
        self._mapping = None
        self._device = "cpu"
        # GPU might be constly
        # self._device = torch.device("cuda" if \
        #                     torch.cuda.is_available() else "cpu")
        
    def setup_model(self, path_to_model):
        model_class = XLNetForQuestionAnswering
        tokenizer_class = XLNetTokenizer
        self._model = model_class.from_pretrained(path_to_model)
        self._tokenizer = tokenizer_class.from_pretrained("xlnet-base-cased")
        self._model.to(self._device)

    def preprocess_data(self, path_to_input):
        examples = read_squad_examples(input_file=path_to_input, 
                                        is_training=False, 
                                        version_2_with_negative=False)
        features = convert_examples_to_features(examples=examples, 
                                                tokenizer=self._tokenizer,
                                                max_seq_length=192,
                                                doc_stride=128,
                                                max_query_length=64,
                                                is_training=False)
    
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)
        return examples, features, dataset

    def to_list(self, tensor):
        return tensor.detach().cpu().tolist()

    def inference(self, dataset, example, feature):
        eval_dataloader = DataLoader(dataset, batch_size=1)
        print("Coding: test ", len(eval_dataloader))
        all_results = []
        for batch in eval_dataloader:
            print("Coding: ", len(batch))
            print("Coding: batch ", batch)
            
            self._model.eval()
            # batch =  tuple(t.to(self._device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2], 
                          # 'example_indices':  batch[3],
                         'cls_index': batch[4],
                         'p_mask':    batch[5]}
                example_indices = batch[3]
                outputs = self._model(**inputs)
        print("INFO: result ", len(outputs))
        print("example_indices ", example_indices)
        i = 0
        for i, example_index in enumerate(example_indices):
            eval_feature = feature[example_index.item()]
            unique_id = int(eval_feature.unique_id)
        
            result = RawResultExtended(unique_id            = unique_id,
                                       start_top_log_probs  = self.to_list(outputs[0][i]),
                                       start_top_index      = self.to_list(outputs[1][i]),
                                       end_top_log_probs    = self.to_list(outputs[2][i]),
                                       end_top_index        = self.to_list(outputs[3][i]),
                                       cls_logits           = self.to_list(outputs[4][i]))
            print("Coding: ", result)
            all_results.append(result)
        return all_results
            

    def postprocess(self, all_examples, all_features, all_results):
        start_n_top = self._model.config.start_n_top 
        end_n_top = self._model.config.end_n_top
        max_answer_length = 30 
        n_best_size = 20
        print("COding: start end ", start_n_top, end_n_top)
        _PrelimPrediction = collections.namedtuple( "PrelimPrediction",
                                ["feature_index", "start_index", "end_index",
                                 "start_log_prob", "end_log_prob"])
        example_index_to_features = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            print("Coding: result ", result)
            unique_id_to_result[result.unique_id] = result

        for (example_index, example) in enumerate(all_examples):
            print("Coding: a b ", example_index, examples)
            # features = example_index_to_features[example_index]

            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            score_null = 1000000  # large and positive

            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
          
                cur_null_score = result.cls_logits

                # if we could have irrelevant answers, get the min score of irrelevant
                score_null = min(score_null, cur_null_score)
                for i in range(start_n_top):
                    for j in range(end_n_top):
                        start_log_prob = result.start_top_log_probs[i]
                        start_index = result.start_top_index[i]

                        j_index = i * end_n_top + j

                        end_log_prob = result.end_top_log_probs[j_index]
                        end_index = result.end_top_index[j_index]

                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= feature.paragraph_len - 1:
                            continue
                        if end_index >= feature.paragraph_len - 1:
                            continue

                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue

                        prelim_predictions.append(
                                                  _PrelimPrediction(
                                                  feature_index=feature_index,
                                                  start_index=start_index,
                                                  end_index=end_index,
                                                  start_log_prob=start_log_prob,
                                                  end_log_prob=end_log_prob))

            prelim_predictions = sorted(prelim_predictions,
                                        key=lambda x: (x.start_log_prob + x.end_log_prob),
                                        reverse=True)
            print("Coding: prelim_predictions ", prelim_predictions )
            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= n_best_size: break
                feature = features[pred.feature_index]

            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = self._tokenizer.convert_tokens_to_string(tok_tokens)

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)
            verbose_logging = False
            final_text = get_final_text(tok_text, orig_text, 
                                        self._tokenizer.do_lower_case, 
                                        verbose_logging) 
            print("Coding: final text ", final_text)
            if final_text in seen_predictions: continue
            seen_predictions[final_text] = True
            """
          """ 


    def handle(self, path_to_input, data, context):
        if not _service.initialized:
            _service.initialize(context)
    
        if data is None: return None

        examples, features, dataset = self.preprocess_data(path_to_input)
    
        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)
    
        return data

path_to_data = "interence.json"
meq_qa_inference = MedQAInference()
meq_qa_inference.setup_model("./bestmodels/BioASQ_Jan292020/outdir/")
examples, features, dataset =  meq_qa_inference.preprocess_data(path_to_data)
all_results = meq_qa_inference.inference(dataset, examples, features)
meq_qa_inference.postprocess(examples, features, all_results)
