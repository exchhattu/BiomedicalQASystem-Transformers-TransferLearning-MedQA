"""
Author: Rojan Shresthat, Insight Data Science 
Tue Jan 21 16:18:39 2020
"""

from pytorch_transformers import XLNetTokenizer

from Data import MedicalData 

class EDA:

    def __init__(self, argument_parser):
        self._dataset = {}
        self._qa_data = {}

        self._tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
        
        t_files = str(argument_parser).split(",")
        for s_file in t_files:
            t_parts = s_file.split(":")
            if len(t_parts) == 2:
                self._dataset[t_parts[0]] = t_parts[1]
            else:
                print("Warn: invalid input format.")
        self._parse_file()

    def _parse_file(self):
        """
        parse the given json file in order to get statistis
        multiple file can be inputted. However, each file should be 
        differntiated by their unique name.
        """

        # Collect all data
        o_mdata     = MedicalData()
        t_fnames    = self._dataset.keys()
        for s_data_name in t_fnames:
            s_fpath = self._dataset[s_data_name] 
            num_qas = o_mdata.parse_json_file(s_fpath, True)
            self._qa_data[s_data_name] = o_mdata._qa_pair 

        for s_data_name in self._qa_data.keys():
            t_qa_pair = self._qa_data[s_data_name] 
            for s_qa_pair in t_qa_pair:
                query_tokens = self._tokenizer.tokenize(s_qa_pair._question)
                s_qa_pair.update_question(len(s_qa_pair._question), len(query_tokens))

                query_tokens = self._tokenizer.tokenize(s_qa_pair._context)
                s_qa_pair.update_context(len(s_qa_pair._context), len(query_tokens))

                query_tokens = self._tokenizer.tokenize(s_qa_pair._answer)
                s_qa_pair.update_answer(len(s_qa_pair._answer), len(query_tokens))

    # def _compute_sequencec_length(self):
