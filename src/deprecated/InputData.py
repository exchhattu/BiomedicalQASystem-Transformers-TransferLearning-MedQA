"""
Rojan Shrestha, Insight Data Science
Wed Jan 22 07:11:37 2020
"""

import os
import json
import numpy as np

from squad_update import SquadResult, SquadV1Processor, SquadV2Processor

# from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers import (WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    # RobertaConfig,
    # RobertaForQuestionAnswering,
    # RobertaTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
    )

import collections


class InputData:

    def __init__(self, s_path_to_dir):
        """
        Tokenize the datasets and convert into numeric value 
        for EDA.
        """
        self._dataset = {}
        self._qa_data = {}

        # SquadV1Processor does not hold start_pos in object
        # self._d_start_pos = collections.OrderedDict()

        t_files =  os.listdir(s_path_to_dir)
        for s_file_name in t_files: 
            if s_file_name.endswith(".json"): 
                st_bname = os.path.basename(s_file_name)
                # s_path_to_file = os.path.join(s_path_to_dir, s_file_name) 
                self._dataset[st_bname] = s_path_to_dir 
        self._parse_file()

        
    def _parse_file(self):
        """
        parse the given json file in order to get statistis
        multiple file can be inputted. However, each file should be 
        differntiated by their unique name.
        """
        # Squaad V1 format - no control over separating for training 
        # testing ....
        processor = SquadV1Processor()

        # Collect all data
        t_fnames    = self._dataset.keys()
        for s_data_name in t_fnames:
            s_fpath = self._dataset[s_data_name] 
            print("INFO: parsing %s ..." %s_fpath)

            examples = processor.get_train_examples(s_fpath, filename=s_data_name)
            print("Coding: ", len(examples))
            self._qa_data[s_data_name] = examples 

    def merge_and_split_data(self, ratio="8:1:1", write_file=False):
        """
        Merge a data collected from multiple files and divide
        subsequently into train, test, and validation randomly


        """
        self._train_examples = []
        self._test_examples = []
        self._valid_examples = []

        all_examples = []
        for example in self._qa_data.values():
            all_examples += example
            print("INFO: after merging %d new dim. %d " %(len(example), len(all_examples)))
      
        t_ratio = ratio.split(":") 
        # split train, valid, test 
        n_samples = len(all_examples)
        a_random_idx = np.arange(n_samples)
        np.random.shuffle(a_random_idx)
        n_train = int(n_samples * float(t_ratio[0])/10.0)  
        n_valid = int(n_samples * float(t_ratio[1])/10.0)  
        n_test  = int(n_samples * float(t_ratio[2])/10.0)  

        self._train_examples = np.array(all_examples)[a_random_idx[:n_train]]
        self._valid_examples = np.array(all_examples)[a_random_idx[n_train+1:n_train+n_valid]]
        self._test_examples = np.array(all_examples)[a_random_idx[-n_test:]]
        print("INFO: train data %d (%0.2f)" %(len(self._train_examples), 10.00*float(t_ratio[0])))
        print("INFO: valid data %d (%0.2f)" %(len(self._valid_examples), 10.00*float(t_ratio[1])))
        print("INFO: test data %d (%0.2f)" %(len(self._test_examples), 10.00*float(t_ratio[2])))

        if write_file:
            self._write_json_file(self._train_examples, file_name = "train.json")
            self._write_json_file(self._valid_examples, file_name = "valid.json")
            self._write_json_file(self._test_examples, file_name = "test.json")


    def _write_json_file(self, examples, file_name = "predict.json"):
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
            # print(example.doc_tokens)

            d_answer = {'text': example.answer_text, 
                        'answer_start' : example.start_position_character }
            # changed qas_id to id since original file format contains id rather than qas_id
            d_sub_data = { "id": example.qas_id,
                           "question": example.question_text,
                           "answers": [d_answer],
                            }
            d_data = {"qas": [d_sub_data], "context": example.context_text}
            t_data.append(d_data)
        d_paragraph = {"paragraphs": t_data} 
        d_final_data = {"data": [d_paragraph]}
        
        # Save as a JSON file
        full_path = os.path.join(os.getcwd(), file_name)
        with open(full_path, 'w') as f:
            json.dump(d_final_data, f)
