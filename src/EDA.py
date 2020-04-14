"""
Author: Rojan Shresthat, Insight Data Science 
Tue Jan 21 16:18:39 2020
"""
import argparse
import os

from pytorch_transformers import XLNetTokenizer
from PyTorch_transform_wrapper import PyTorchTransformWrapper


class EDA:
    def __init__(self, s_path_to_dir):

        """
        Tokenize the datasets and convert into numeric value 
        for EDA.
        """
        self._dataset = {}
        self._qa_data = {}

        self._tokenizer = XLNetTokenizer.from_pretrained(
            "xlnet-base-cased", do_lower_case=True
        )

        t_files = os.listdir(s_path_to_dir)
        for s_file_name in t_files:
            if s_file_name.endswith(".json"):
                st_bname = os.path.basename(s_file_name)
                s_path_to_file = os.path.join(s_path_to_dir, s_file_name)
                self._dataset[st_bname] = s_path_to_file

        self._parse_file()
        self._write_file(path_to_file="bioasq_eda.data")

    def _parse_file(self):
        """
        parse the given json file in order to get statistis
        multiple file can be inputted. However, each file should be 
        differntiated by their unique name.
        """

        t_fnames = self._dataset.keys()
        for s_data_name in t_fnames:
            s_fpath = self._dataset[s_data_name]
            print("INFO: parsing %s ..." % s_fpath)
            o_pytorch = PyTorchTransformWrapper()
            o_pytorch.read_squad_formatted_json_file(s_fpath)
            self._qa_data[s_data_name] = o_pytorch._qa_data

    def _write_file(self, path_to_file="bioasq_eda.data"):
        """
        Write a file that contains distribution of question,
        answer, and context length 

        params:
            path_to_file: output file
        """
        fpath_to_file = os.path.join(os.getcwd(), path_to_file)
        with open(fpath_to_file, "w") as of:
            s_line = "id,question_len,num_word_in_question,answer_len,num_word_in_answer,num_word_in_token\n"
            for s_data_name in self._qa_data.keys():
                t_qa_pair = self._qa_data[s_data_name]
                for s_qa_pair in t_qa_pair:
                    s_line += "%s," % s_qa_pair.qas_id

                    # question
                    query_tokens = self._tokenizer.tokenize(s_qa_pair.question_text)
                    s_line += "%s,%s," % (
                        len(s_qa_pair.question_text),
                        len(query_tokens),
                    )

                    # answer
                    query_tokens = self._tokenizer.tokenize(s_qa_pair.orig_answer_text)
                    s_line += "%s,%s," % (
                        len(s_qa_pair.orig_answer_text),
                        len(query_tokens),
                    )

                    # context
                    s_line += "%d" % (len(s_qa_pair.doc_tokens))
                    of.write(s_line)
                    of.write("\n")
                    s_line = ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eda",
        default=None,
        type=str,
        required=True,
        help="analyze a training, validation, and test data, example train:filename, test:filename, validate:filename",
    )

    args = parser.parse_args()
    if args.eda:
        eda = EDA(args.eda)
