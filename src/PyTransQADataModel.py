
"""
https://github.com/huggingface/transformers/blob/master/examples/run_squad.py
"""
import os
import json
import collections
import random
import numpy as np

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from pytorch_transformers import XLNetConfig, XLNetModel, XLNetTokenizer, XLNetForQuestionAnswering
from pytorch_transformers.optimization import WarmupConstantSchedule
from pytorch_transformers import AdamW
from utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended)
from utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad



from transformers.data.processors.squad import SquadResult
from transformers.data.metrics import squad_metrics
from transformers import get_linear_schedule_with_warmup

from tqdm import tqdm, trange

class QaDataModel:

    """
    For training and testing a given data
    """

    def __init__(self):
        self.init_params('xlnet', 'xlnet-base-cased', f_lr=5e-5, f_eps = 1e-8)

        self._max_seq_length      = 384
        self._doc_stride          = 128
        self._max_query_length    = 64

    def _create_dataset(self, examples, evaluate=False, output_examples=False):
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=self._tokenizer,
                                                max_seq_length=self._max_seq_length,
                                                doc_stride=self._doc_stride,
                                                max_query_length=self._max_query_length,
                                                is_training=not evaluate)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

        if evaluate:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_example_index, all_cls_index, all_p_mask)
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_start_positions, all_end_positions,
                                    all_cls_index, all_p_mask)

        if output_examples:
            return dataset, examples, features
        return dataset

    def init_params(self, model_name, pre_trained_model, f_lr=5e-5, f_eps = 1e-8):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MODEL_CLASSES   = { "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer) }
        # self._config, self._model_class, self._tokenizer = MODEL_CLASSES[model_name]
        self._tokenizer = XLNetTokenizer.from_pretrained(pre_trained_model, do_lower_case=True)
        self._config    = XLNetConfig.from_pretrained(pre_trained_model, do_lower_case=True)
        self._model     = XLNetForQuestionAnswering.from_pretrained(pre_trained_model, config=self._config)

        no_decay = ['bias', 'LayerNorm.weight']
        weight_decay = 0.0 # Author's default parameter
        optimizer_grouped_parameters = [
                  {'params': [p for n, p in self._model.named_parameters() \
                          if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                  {'params': [p for n, p in self._model.named_parameters() \
                          if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                  ]
        # warmup_steps = 0.0
        self._optimizer = AdamW(optimizer_grouped_parameters, lr=f_lr, eps=f_eps)

    def do_end_to_end(self, train_data, valid_data):
        """
        train a pretrain model using medical data (domain specific data) end to end.

        params:
            train_example: training set
            valid_example: validation set
    
        """
        self.set_seed()
        train_dataset = self._create_dataset(train_data, evaluate=False)

        valid_dataset, valid_example, valid_feature = \
                self._create_dataset(valid_data, evaluate=True, output_examples=True)
        self.write_json_file(valid_data, file_name = "predict.json")

        train_batch_size    = 8 # args.per_gpu_train_batch_size * max(1, args.n_gpu) # taken from ...
        train_sampler       = RandomSampler(train_dataset) 
        train_dataloader    = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        max_step    = 1 # default value in run_squad.py
        t_total     = max_step 
        gradient_accumulation_steps = 1 # default value in run_squad.py
        self._scheduler = get_linear_schedule_with_warmup(self._optimizer, 
                                                          num_warmup_steps=0 , # default value in run_squad.py 
                                                          num_training_steps=t_total) 

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        self._model.zero_grad()
        # set random seed.
        epochs = 2 
        epoch = 0
        for _ in trange(epochs, desc="Iteration"):
            epoch += 1
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self._model.train()
                batch = tuple(t.to(self._device) for t in batch)

                inputs = {'input_ids':       batch[0],
                          'attention_mask':  batch[1], 
                          'token_type_ids':  batch[2],  
                          'start_positions': batch[3], 
                          'end_positions':   batch[4],
                          'cls_index':       batch[5],
                          'p_mask':          batch[6] }
                outputs = self._model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0) # max_grad_norm = 1.0 by default
                tr_loss += loss.item()
                print("Coding: loss ", epoch , step, tr_loss, loss.item(), loss.mean(), len(batch))
                
                # this function does not have effect unless gradient_accumulation_steps > 1
                # however, it was set to 1 in original code.
                # if (step + 1) % 1 == 0:

                self._optimizer.step()
                self._scheduler.step()  # Update learning rate schedule
                self._model.zero_grad()
                global_step += 1

                # validation
                result = self.evaluate(valid_dataset, valid_example, 
                                       valid_feature, self._model, 
                                       self._tokenizer, prefix="")
                print("Coding: result ", epoch, result)

        # write something for checkpoint 
        # check points
        self._write_check_points(self._model, global_step)

        # 

    def write_json_file(self, examples, file_name = "predict.json"):
        """
        make json format of qa data and dump into a given file path

        param:
            examples: example file to be written 
            file_name: file name  
            is_debug: for testing purpose
        """
        # create a dictionary to dump as json 
        d_final_data = collections.OrderedDict()
        d_data = collections.OrderedDict()
        t_data = []
        for example in examples:
            d_sub_data = collections.OrderedDict()
            d_answer = collections.OrderedDict() 

            d_answer = {'text': example.orig_answer_text, 
                        'answer_start' : example.start_position }
            # changed qas_id to id since original file format contains id rather than qas_id
            d_sub_data = { "qas_id": example.qas_id,
                           "question": example.question_text,
                           "answers": [d_answer],
                            }
            d_data = {"qas": [d_sub_data] }
            t_data.append(d_data)
        d_paragraph = {"paragraphs": t_data} 
        d_final_data = {"data": [d_paragraph]}
        
        # Save as a JSON file
        full_path = os.path.join(os.getcwd(), file_name)
        with open(full_path, 'w') as f:
            json.dump(d_final_data, f)

    def set_seed():
        """ to reproduce the result  """

        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _to_list(self, tensor):
        """ helper function """
        return tensor.detach().cpu().tolist()

    def predict_using_predefined_models(self, test_data):
        """
        predict a test data using trained model

        params:
            test_data: test data to check a performance of a model
        """
        test_dataset, test_example, test_feature = \
                self._create_dataset(test_data, evaluate=True, output_examples=True)

        self.write_json_file(test_data, file_name = "predict.json")

        # has to dump json file  
        global_step = 1 
        self.evaluate(test_dataset, test_example, test_feature,
                        self._model, self._tokenizer, prefix=global_step)

    def evaluate(self, dataset, examples, features, model, tokenizer, prefix=1):
        # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_batch_size = 8 # by default # args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset) # if args.local_rank == -1 else DistributedSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)
        s_pred_file = os.path.join(os.getcwd(), "predict.json")

        all_results = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(self._device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],  
                          'cls_index':      batch[4],
                          'p_mask':         batch[5]
                         }
                example_indices = batch[3]
                outputs = model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                # XLNet uses a more complex post-processing procedure
                result = RawResultExtended(unique_id = unique_id,
                                            start_top_log_probs  = self._to_list(outputs[0][i]),
                                            start_top_index      = self._to_list(outputs[1][i]),
                                            end_top_log_probs    = self._to_list(outputs[2][i]),
                                            end_top_index        = self._to_list(outputs[3][i]),
                                            cls_logits           = self._to_list(outputs[4][i]))
                all_results.append(result)

        # Compute predictions
        output_dir = os.getcwd()
        output_prediction_file = os.path.join(output_dir, "predictions_{}.json".format(prefix))
        output_nbest_file = os.path.join(output_dir, "nbest_predictions_{}.json".format(prefix))
        version_2_with_negative = True
        output_null_log_odds_file = None
        if version_2_with_negative:
            output_null_log_odds_file = os.path.join(output_dir, "null_odds_{}.json".format(prefix))
        else:
            output_null_log_odds_file = None

        # XLNet uses a more complex post-processing procedure
        n_best_size = 20
        max_answer_length = 30
        verbose_logging = True
        predict_file = "predict.json"
        write_predictions_extended(examples, features, all_results, n_best_size,
                                    max_answer_length, output_prediction_file,
                                    output_nbest_file, output_null_log_odds_file, predict_file,
                                    model.config.start_n_top, model.config.end_n_top,
                                    version_2_with_negative, tokenizer, verbose_logging)

        # Evaluate with the official SQuAD script
        evaluate_options = EVAL_OPTS(data_file=s_pred_file,
                                    pred_file=output_prediction_file,
                                    na_prob_file=output_null_log_odds_file)
        results = evaluate_on_squad(evaluate_options)
        # print("Coding: final result ", results)
        return results

    def _write_check_points(self, model, step_num):
        path_to_output_dir = os.path.join(os.getcwd(), "checkpts")
        output_dir = os.path.join(path_to_output_dir, 'checkpoint-{}'.format(step_num))
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        # model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model.save_pretrained(output_dir)
        torch.save(output_dir, os.path.join(output_dir, 'training_args.bin'))
        print("Saving model checkpoint to %s", output_dir)

    def serve_models(self, test_data, path_to_model):
        path_to_model = "./checkpts/checkpoint-16/" 
        test_dataset, test_example, test_feature = \
                self._create_dataset(test_data, evaluate=True, output_examples=True)
        self.write_json_file(test_data, file_name = "predict.json")
        model_class = XLNetForQuestionAnswering
        self._model = model_class.from_pretrained(path_to_model)
        self._model.to(self._device)
        self.evaluate(test_dataset, test_example, test_feature, 
                        self._model, self._tokenizer, prefix="")

