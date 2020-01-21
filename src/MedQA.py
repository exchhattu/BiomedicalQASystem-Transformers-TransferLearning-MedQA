"""
Author: Rojan Shresthat, Insight Data Science 
Tue Jan 21 16:04:06 2020
"""

import argparse
import logging
import os

from EDA import EDA 


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    ## Required parameters
    # parser.add_argument("--mode", default="transfer_model", type=str, required=True,
    #                     help="select one from (transfer_model, pretrained, fronzen)")
    # parser.add_argument("--pretrained", default="XLnet", type=str, required=True,
    #                     help="XLNet models or others")
    # parser.add_argument("--test", default="None", type=str, required=True,
    #                     help="path to test data")

    # optional data
    parser.add_argument("--eda", default=None, type=str, required=False,
            help="analyze a training, validation, and test data, example train:filename, test:filename, validate:filename")

    args = parser.parse_args()
    if args.eda:
        eda = EDA(args.eda)

