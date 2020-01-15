'''
Worked @ Insight Data Science
Wed Jan 15 10:55:41 2020
'''
import xml.dom.minidom
import xml.etree.ElementTree as etree 

class MedicalData: 

    class QuestionPair:
        def __init__(self, id, question, answer):
            m_questions = {} 
            m_answers   = {} 


    def __init__(self):

    def parse_xml_input(self, s_path_to_file):
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
        
        # create empty list for news items 
        newsitems = [] 
                                   
        # iterate news items 
        for item in o_root.findall('QAPairs'):
            print(item)

