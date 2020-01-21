'''
Worked @ Insight Data Science
Wed Jan 15 10:55:41 2020
''' 

# python
import os

# parser
import json 
import xml.dom.minidom
import xml.etree.ElementTree as etree 

# data science/AI
import pandas as pd
from DataEncode import InputData 
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)


class MedicalData: 

    class QuestionPair:
        def __init__(self, id, question, answer, qtype, keyword):
            self._id = id 
            self._question  =  question 
            self._answer    =  answer 
            self._qtype = qtype 
            self._keyword = keyword

        def put_context(self, answer_start, context, is_impossible):
            """
            set context and corresponding answer position 
            params:
                answer_start: position of answer in context
                context: context text mostly paragraph
            """
            self._answer_start = answer_start
            self._context = context
            self._is_impossible = is_impossible

        def update(self, length, start_position, end_position, doc_tokens):
            """
            Udpate some values after parsing paragraph
            def update(self, length, end_length, docs):
            """
            self._length = length
            self._start_position = start_position 
            self._end_position = end_position
            self._doc_tokens = doc_tokens

    def __init__(self):
        self._qa_pair = [] # list of question answer pair 
        self._test_conds = {} 

    def test(self):
        idata = InputData()
        t_features = []
        for o_qa in self._qa_pair:
            idata.clean_input(o_qa)
            t_features.append(idata.tokenize_input(o_qa))
            break
        return self._convert_features_to_dataset(t_features)

    def _convert_features_to_dataset(self, t_features): 
        t_input_ids = []
        t_input_masks = []
        t_segment_ids = []
        t_cls_idxs = []
        t_p_mask = []
        t_st_pos = []
        t_ed_pos = []

        for o_feature in t_features: 
            f_input_id, f_input_mask, f_segment_id, f_cls_idx, f_p_mask, f_st_pos, f_ed_pos = o_feature 
            t_input_ids.append(f_input_id)
            t_input_masks.append(f_input_mask)
            t_segment_ids.append(f_segment_id)
            t_cls_idxs.append(f_cls_idx)
            t_p_mask.append(f_p_mask)
            t_st_pos.append(f_st_pos)
            t_ed_pos.append(f_ed_pos)

        te_input_ids    = torch.tensor(t_input_ids, dtype=torch.long)
        te_input_masks  = torch.tensor(t_input_masks, dtype=torch.long)
        te_segment_ids  = torch.tensor(t_segment_ids, dtype=torch.long)
        te_cls_idx      = torch.tensor(t_cls_idxs, dtype=torch.long)
        te_p_mask       = torch.tensor(t_p_mask, dtype=torch.float)
        te_st_pos       = torch.tensor(t_st_pos, dtype=torch.long)
        te_ed_pos       = torch.tensor(t_ed_pos, dtype=torch.long)

        dataset = TensorDataset(te_input_ids, te_input_masks, te_segment_ids,
                                te_st_pos, te_ed_pos, te_cls_idx, te_p_mask)
        return dataset


    def dissect(self):
        print("# of qa pairs: %d" %len(self._qa_pair))

    def parse_xmls(self, s_path_to_dir, is_debug=False):
        """
        scan all parse XML files from directory and pass
        them for parsing

        params:
            s_path_to_dir: path to a directory
            is_debug: allow debug mostly testing
        """
       
        i_counter= 0 
        t_files =  os.listdir(s_path_to_dir)
        for s_file_name in t_files: 
            if s_file_name.endswith(".xml"): 
                s_path_to_file = os.path.join(s_path_to_dir, s_file_name) 
                self.parse_xml_input(s_path_to_file, is_debug)
                i_counter += 1 
        print("INFO: %d files was/were parsed." %i_counter)
        if is_debug: return i_counter



    def parse_xml_input(self, s_path_to_file, is_debug=False):
        """
        parse XML file, create a dictionary to hold question 
        answer pairing

        params:
          s_path_to_file: path to XML file
        """

        # create element tree object 
        o_xml_tree = etree.parse(s_path_to_file) 
        # get root element 
        o_root = o_xml_tree.getroot() 
       

        q_focuses = o_root.findall('Focus')
        s_focus = ""
        for q_focus in q_focuses: 
            s_focus = s_focus + ";" + q_focus.text
        # iterate news items 
        for q_pairs in o_root.findall('QAPairs'):
            for q_pair in q_pairs: # QApair  
                for q_data in  q_pair: 
                    s_question, s_answer, s_qtype = "", "", ""
                    if q_data.tag == "Question":
                        s_question = q_data.text 
                    elif q_data.tag == "Answer":
                        s_answer= q_data.text 

                    # id and question type are attributes. just keep question type.
                    for attrib in q_data.attrib.keys():
                        if attrib == "qtype":
                            s_qtype = q_data.attrib[attrib]
                    o_qa = self.QuestionPair("", s_question, s_answer, s_qtype, s_focus)
                    self._qa_pair.append(o_qa)

        if is_debug:
            self.dissect()

    def parse_excel_file(self, path_to_file, is_debug=False):
        """
        read a excel file in order 

        args:
            path_to_file: path to the excel file
            is_debug: decide whether debug or not.
                      For unittest, it should be turn True
        """

        try:
            o_excel_data = pd.ExcelFile(path_to_file)
            t_sheets = o_excel_data.sheet_names
            if "DrugQA" in t_sheets:
                df_drug_qa = o_excel_data.parse("DrugQA")
                self.parse_drug_qa_from_excel(df_drug_qa, is_debug)
            elif "QS" in o_excel_data.sheet_names:
                print("QS is not included")
            else:
                print("[WARN] not found excel sheet")

        except IOError:
            print("[WARN]: not found %s" %path_to_file) 

    def parse_drug_qa_from_excel(self, df_drug_qa, is_debug = False):
        """
        parse a data frame read from excel file 

        args:
            df_drug_qa: dataframe containing the drug related 
                        question answering 
        """
        for index, row in df_drug_qa.iterrows():
            o_qa = self.QuestionPair("", row['Question'], 
                                     row['Answer'], 
                                     row['Question Type'], 
                                     row['Focus (Drug)'])
            self._qa_pair.append(o_qa)
            if is_debug and index == 5: 
                self._test_conds['excel_parser'] = o_qa 

    def extract_put_qas(self, qas):
        """
        parse nested dictionary 

        params: 
            qas: question answer data
        """
        o_qa        = qas["qas"][0]
        s_id        = o_qa["id"]
        s_question  = o_qa["question"]
        # answers has two keys - text and answer_start in the context
        o_answers       = o_qa["answers"]
        s_answer        = o_answers[0]["text"]
        i_answer_start  = o_answers[0]["answer_start"]

        # next key of qas is context
        s_context       = qas["context"]

        # create question answer object
        o_qap = self.QuestionPair(s_id, s_question, s_answer, "", "")
        o_qap.put_context(i_answer_start, s_context, False)
        self._qa_pair.append(o_qap) 

    def parse_json_file(self, path_to_file, is_debug=False):
        """
        parse json file especially from BioASQ

        params:
            path_to_file: path to json file
            is_debug: turn on while testing or development
        """

        try:
            with open(path_to_file) as oj_fjson:
                dc_data = json.load(oj_fjson)
            oj_fjson.close()
            
            # Looking smart way to parse
            d_para = dc_data['data'][0]
            # qas and context are extracted.
            for d_qass in d_para['paragraphs']:
                self.extract_put_qas(d_qass)
            if is_debug: return len(self._qa_pair)
        except IOError:
            print("[WARN]: not found %s" %path_to_file) 

    def get_answer(self, qa_pair):
        """
        get answer for qa_pair

        params:
            qa_pair: qa_pair object 
        """
        # answer
        answers = []
        d_answer = {'text':  qa_pair._answer,  
                    'answer_start': qa_pair._answer_start}
        answers.append(d_answer)
        return answers

    def get_question(self, qa_pair):
        """
        get quesion with other properties such as id,
        is_impossible

        param:
            qa_pair: object of class QuestionPair
        """
        t_qas = []
        d_qas = { 'id': "NA",
                  'is_impossible': False,
                  'question': qa_pair._question, 
                  'answer': self.get_answer(qa_pair)
                }
        t_qas.append(d_qas)
        return t_qas


    def write_json_file(self, path_to_file, file_name = "train.json", is_debug=False):
        """
        make json format of qa data and dump into a given file path

        param:
            path_to_file: path to a file 
            is_debug: for testing purpose
        """
        # create a dictionary to dump as json 
        t_data = []
        for qa_pair in self._qa_pair:
            # qas 
            d_data = { 'context': qa_pair._context, 
                       'qas': self.get_question(qa_pair) }
            t_data.append(d_data)            

        # Save as a JSON file
        full_path = os.path.join(path_to_file, file_name)
        with open(full_path, 'w') as f:
            json.dump(t_data, f)


    def tokenize(self, is_word=True): 
        """
        Sentences is tokenized based on the input given
        There will be two ways to tokenize - word and 
        sentence.

        param:
            is_word - determines how tokenization is applied 
        """


