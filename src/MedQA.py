"""
Author: Rojan Shresthat, Insight Data Science 
Tue Jan 21 16:04:06 2020
"""

import argparse
import logging
import sys, os
import subprocess

from PyTransInputData import InputData 
from QADataModel import QaDataModel

def train_valid_cmd(path_to_train, path_to_test):
    program_path = os.path.join(os.getcwd(), "src/run_squad_update.py") 
    if not os.path.exists(program_path):
        print("FATAL: %s not found" %program_path)
        sys.exit()

    default_args = ( "python3",  program_path, \
              "--overwrite_cache",  "--overwrite_output_dir", "--model_type", "xlnet", 
              "--model_name_or_path", "xlnet-base-cased", "--output_dir", "output_logs" 
              "--do_lower_case",  "--learning_rate", "3e-5",  
              "--num_train_epochs", "5", "--max_seq_length", "192", 
              "--doc_stride", "128", "--train_file", path_to_train,
              "--predict_file", path_to_test, "--do_train", "--do_eval",
              "--num_sample", "10")
    return default_args


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
    data.merge_and_split_data(ratio="9.00:1.00:0.00", write_file=True)
    path_to_train = os.path.join(data._path_to_dir, "train.json") 
    path_to_valid = os.path.join(data._path_to_dir, "valid.json") 
    path_to_test = os.path.join(data._path_to_dir, "test.json") 

    st_cmd = train_valid_cmd(path_to_train, path_to_valid)

    qa_data_model = QaDataModel()
    if args.use_pretrained_model: 
        qa_data_model.predict_using_predefined_models(data._test_examples)
    elif args.end_to_end:
        popen = subprocess.Popen(st_cmd, stdout=subprocess.PIPE)
        # popen.wait()
        # output = popen.stdout.read()
        # qa_data_model.do_end_to_end(data._train_examples, data._valid_examples)


