"""
Author: Rojan Shresthat, Insight Data Science 
Tue Jan 21 16:04:06 2020
"""

import argparse
import logging
import os

from InputData import InputData 
from QADataModel import QaDataModel


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    # parser.add_argument("--mode", default="transfer_model", type=str, required=True,
    #                     help="select one from (transfer_model, pretrained, fronzen)")
    # parser.add_argument("--use_pretrained", default="XLnet", type=str, required=True,
    #                     help="XLNet models or others")
    parser.add_argument("--data", default="None", type=str, 
                            required=True,
                            help="path to data directory")

    args = parser.parse_args()
    data = InputData(args.data)
    # I will be accessible to get EDA
    data.merge_and_split_data(ratio="8:1:1")


    qa_data_model = QaDataModel()
    qa_data_model.predict_using_predefined_models(data._test_examples)

    

