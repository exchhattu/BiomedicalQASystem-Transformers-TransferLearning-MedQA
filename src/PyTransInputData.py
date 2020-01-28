"""
Rojan Shrestha, Insight Data Science
Wed Jan 22 07:11:37 2020
"""

import os
import numpy as np

from pytorch_transformers import XLNetModel, XLNetTokenizer
from pytorch_transformers import AdamW


from PyTorch_transform_wrapper import PyTorchTransformWrapper

import collections


class InputData:

    def __init__(self, s_path_to_dir):
        """
        Tokenize the datasets and convert into numeric value 
        for EDA.
        """
        self._dataset = {}
        self._qa_data = {}

        t_files =  os.listdir(s_path_to_dir)
        for s_file_name in t_files: 
            if s_file_name.endswith(".json"): 
                st_bname = os.path.basename(s_file_name)
                s_path_to_file = os.path.join(s_path_to_dir, s_file_name) 
                self._dataset[st_bname] = s_path_to_file 
        self._parse_file()

        
    def _parse_file(self):
        """
        parse the given json file in order to get statistis
        multiple file can be inputted. However, each file should be 
        differntiated by their unique name.
        """

        # Collect all data
        t_fnames    = self._dataset.keys()
        for s_data_name in t_fnames:
            s_fpath = self._dataset[s_data_name] 
            print("INFO: parsing %s ..." %s_fpath)
            o_pytorch = PyTorchTransformWrapper()
            o_pytorch.read_squad_formatted_json_file(s_fpath) 
            self._qa_data[s_data_name] = o_pytorch._qa_data 

    def merge_and_split_data(self, ratio="8:1:1"):
        """
        merge a data collected from multiple files and divide
        subsequently into train, test, and validation randomly
        """
        self._train_examples = []
        self._test_examples = []
        self._valid_examples = []

        all_examples = []
        for example in self._qa_data.values():
            print("Coding: ", len(example))
            all_examples += example
            print("INFO: after merging %d new dim. %d " %(len(example), len(all_examples)))
            break
      
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
