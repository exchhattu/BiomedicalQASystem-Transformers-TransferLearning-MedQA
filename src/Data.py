'''
Worked @ Insight Data Science
Wed Jan 15 10:55:41 2020
''' 

import pandas as pd

import xml.dom.minidom
import xml.etree.ElementTree as etree 

class MedicalData: 

    class QuestionPair:
        def __init__(self, id, question, answer, qtype, keyword):
            self._id = id 
            self._question  =  question 
            self._answer    =  answer 
            self._qtype = qtype 
            self._keyword = keyword


    def __init__(self):
        self._qa_pair = [] # list of question answer pair 
        self._test_conds = {} 

    def dissect(self):
        print("# of qa pairs: %d" %len(self._qa_pair))



    def parse_xml_input(self, s_path_to_file, is_debug=False):
        """
        parse XML file, create a dictionary to hold question 
        answer pairing

        params:
          s_path_to_file: path to XML file
        """
        # o_xml_doc = xml.dom.minidom.parse(s_path_to_file)

        # create element tree object 
        o_xml_tree = etree.parse(s_path_to_file) 
        # get root element 
        o_root = o_xml_tree.getroot() 
        print(o_root.tag)
        
        # create empty list for news items 
        newsitems = [] 
                                   
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
                    o_qa = self.QuestionPair("", s_question, s_answer, s_qtype, "")
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
                print("Not yet decided")
            else:
                print("not sure ")

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

                    
    def tokenize(self, is_word=True): 
        """
        Sentences is tokenized based on the input given
        There will be two ways to tokenize - word and 
        sentence.

        param:
            is_word - determines how tokenization is applied 
        """


