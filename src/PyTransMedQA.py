"""
Author: Rojan Shresthat, Insight Data Science 
Tue Jan 21 16:04:06 2020
"""

import argparse
import logging
import sys, os

from PyTransInputData import InputData 
from PyTransQADataModel import QaDataModel


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument('--data', 
                        default='None', 
                        type=str, 
                        required=True,
                        help='path to data directory')
    parser.add_argument('--serve_model', 
                        action='store_true', 
                        help='XLNet models or others')
    parser.add_argument('--end_to_end', 
                        action='store_true', 
                        help='Modify the hyperparameter and parameters')

    args = parser.parse_args()
    data = InputData(args.data)
    # this function will generate - data._test_examples, data._train_examples, data._valid_examples
   #  data.merge_and_split_data(ratio="9.00:1.00:0.00", write_file=True)
    data.merge_and_split_data(ratio="0.10:0.01:8.50")
    # sys.exit()

    qa_data_model = QaDataModel()
    if args.serve_model: 
        qa_data_model.serve_models(data._test_examples, "")
    elif args.end_to_end:
        qa_data_model.do_end_to_end(data._train_examples, data._valid_examples)
        qa_data_model.serve_models(data._test_examples, "")


