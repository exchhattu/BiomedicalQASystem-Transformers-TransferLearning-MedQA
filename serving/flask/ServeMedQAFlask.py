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
# from torchvision import transforms
import torch.nn.functional as F

from torch.utils.data import (DataLoader, 
                                RandomSampler, 
                                SequentialSampler,
                                TensorDataset)

from pytorch_transformers import  XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer
from utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended,
                         get_final_text, _compute_softmax)

# from flask import Flask, jsonify, request, render_template
# app = Flask(__name__)

class MedQAInference:

    def __init__(self):
        self._initialized = False 
        self._model = None
        self._device = "cpu"
        # GPU might be costly
        self._device = torch.device("cuda" if \
                                    torch.cuda.is_available() else "cpu")
        
    def initialize(self, path_to_model):
        """
        initialize object

        params:
            path_to_model: pretrain model
        """
        model_class = XLNetForQuestionAnswering
        tokenizer_class = XLNetTokenizer
        self._model = model_class.from_pretrained(path_to_model)
        self._tokenizer = tokenizer_class.from_pretrained("xlnet-base-cased")
        self._model.to(self._device)

    def preprocess_data(self, path_to_stream, input_question):
        """
        preprocess input data

        params:
            path_to_stream: document content 
            input_question: input file
        """
        examples = read_squad_examples(input_stream=path_to_stream, 
                                        is_training=False, 
                                        version_2_with_negative=False,
                                        updated_question=input_question)
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
        """
        Helper function - convert tensor to list
        params:
            tensor: input tensor
        """
        return tensor.detach().cpu().tolist()

    def inference(self, dataset, example, feature):
        """
        Inference

        Params:
            dataset: data for prediction
            example:
            feature: 
        """

        eval_dataloader = DataLoader(dataset, batch_size=1)
        all_results = []
        for batch in eval_dataloader:
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
            all_results.append(result)
        return all_results
            

    def postprocess(self, all_examples, all_features, all_results):
        """
        Postprocessing step. The code is borrowed and refactored.

        Params:
            all_examples: list of examples
            all_features: list of features 
            all_results: list of results
        """
        example_index_to_features = collections.defaultdict(list)
        if not len(all_features) == 1 or not len(all_examples) == 1 or not len(all_results)==1: 
            return None

        feature = all_features[0]
        example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        result = all_results[0]
        unique_id_to_result[result.unique_id] = result

        example_index = 0 
        example = all_examples[0]
        features = example_index_to_features[example_index]
        nbest, prob = self.__prelim_pred(features, unique_id_to_result, all_features, example)
        return nbest[0].text

    def __prelim_pred(self, features, unique_id_to_result, all_features, example):
        """
        Gathered preliminary data and preprocessed 

        Params:
            features: selected feature given id
            unique_id_to_result: unique id
            all_features: list of feature
            example: list 
        """
        start_n_top     = self._model.config.start_n_top 
        end_n_top       = self._model.config.end_n_top
        max_answer_length = 30 

        _PrelimPrediction = collections.namedtuple( "PrelimPrediction",
                                ["feature_index", "start_index", "end_index",
                                 "start_log_prob", "end_log_prob"])

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
                    start_log_prob  = result.start_top_log_probs[i]
                    start_index     = result.start_top_index[i]
                    j_index         = i * end_n_top + j
                    end_log_prob    = result.end_top_log_probs[j_index]
                    end_index       = result.end_top_index[j_index]

                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= feature.paragraph_len - 1: continue
                    if end_index >= feature.paragraph_len - 1: continue
                    if not feature.token_is_max_context.get(start_index, False): continue
                    if end_index < start_index: continue
                    length = end_index - start_index + 1
                    if length > max_answer_length: continue

                    prelim_predictions.append( _PrelimPrediction(
                                                    feature_index=feature_index,
                                                    start_index=start_index,
                                                    end_index=end_index,
                                                    start_log_prob=start_log_prob,
                                                    end_log_prob=end_log_prob))

        prelim_predictions = sorted(prelim_predictions,
                                    key=lambda x: (x.start_log_prob + x.end_log_prob),
                                    reverse=True)
        nbest = self.__token_parsing(prelim_predictions, all_features, example)
        probs = self.__get_nbest_prob(nbest)
        return nbest, probs

    def __token_parsing(self, prelim_predictions, all_features, example):
        """
        Parse the preliminary predicated result

        Params:
            prelim_predictions: tuples
            all_features: features of predicted content
            example: selected feature 
        """
        n_best_size = 20
        seen_predictions = {}
        nbest = []
        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                    "NbestPrediction", ["text", "start_log_prob", "end_log_prob"])
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size: break
            feature = all_features[pred.feature_index]
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
            final_text = get_final_text(tok_text, orig_text, self._tokenizer.do_lower_case, verbose_logging) 
            if final_text in seen_predictions: continue
            seen_predictions[final_text] = True
            nbest.append(_NbestPrediction(text=final_text,
                                          start_log_prob=pred.start_log_prob,
                                          end_log_prob=pred.end_log_prob))
            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(_NbestPrediction(text="", start_log_prob=-1e6, end_log_prob=-1e6))

        return nbest

    def __get_nbest_prob(self, nbest):
        """
        Get probabilities of n best prediction, which is subsequently used in
        softmax calculation

        Params:
            nbest: list with n best values
        """
        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry
        probs = _compute_softmax(total_scores)
        return probs
