"""
Author: Rojan Shresthat, Insight Data Science 
Tue Jan 21 16:04:06 2020
"""

import argparse
import logging
import sys, os

from InputData import InputData 
from QADataModel import QaDataModel


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument('--data', default='None', type=str, 
                            required=True,
                            help='path to data directory')
    # parser.add_argument("--mode", default="transfer_model", type=str, required=True,
    #                     help="select one from (transfer_model, pretrained, fronzen)")
    parser.add_argument('--use_pretrained_model', action='store_true', 
                        help='XLNet models or others')
    parser.add_argument('--end_to_end', action='store_true', 
                        help='Modify the hyperparameter and parameters')

    args = parser.parse_args()
    data = InputData(args.data)
    # this function will generate - data._test_examples, data._train_examples, data._valid_examples
    data.merge_and_split_data(ratio="9.0:0.5:0.5", write_file=True)
    sys.exit()

    qa_data_model = QaDataModel()
    if args.use_pretrained_model: 
        qa_data_model.predict_using_predefined_models(data._test_examples)
    elif args.end_to_end:
        print("Coding: I entered ")
        qa_data_model.do_end_to_end(data._train_examples, data._valid_examples)


