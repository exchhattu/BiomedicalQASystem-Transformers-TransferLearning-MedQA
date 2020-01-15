'''
Worked @ Insight Data Science
Wed Jan 15 10:55:41 2020
'''
import xml.dom.minidom
import xml.etree.ElementTree as etree 

class MedicalData: 

    class QuestionPair:
        def __init__(self, id, question, answer, qtype):
            self._id = id 
            self._questions =  question 
            self._answers   =  answer 
            self._qtype = qtype 


    def __init__(self):
        self._qa_pair = [] # list of question answer pair 

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
                    o_qa = self.QuestionPair("", s_question, s_answer, s_qtype)
                    self._qa_pair.append(o_qa)

        if is_debug:
            self.dissect()
                    
    def tokenize(self, is_word=True): 
        """
        Sentences is tokenized based on the input given
        There will be two ways to tokenize - word and 
        sentence.

        param:
            is_word - determines how tokenization is applied 
        """


