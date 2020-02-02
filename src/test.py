'''
Rojan Shrestha
@ Insight Data Science
Wed Jan 15 11:38:19 2020

Test cases
'''
import os

from Data import MedicalData 
from Data import InputData 
from QADataModel import QaDataModel

def test_xml_parser():
    o_mdata = MedicalData()
    # parsing xml file
    o_mdata.parse_xml_input("./unittest/data/0000001.xml", \
                            is_debug=True)

    icount = o_mdata.parse_xmls("./unittest/data/", True)
    assert icount == 3
    print("***Passed XML parser***")

    # parsing excel file
    o_mdata.parse_excel_file("./unittest/data/test_MedInfo2019-QA-Medications.xlsx", True)

    # result
    o_qa_result = o_mdata._test_conds['excel_parser'] 

    # truth
    ## Case1
    s_question_truth = "what is desonide ointment used for"
    assert o_qa_result._question == s_question_truth

    ## Case2
    s_begin_answer_truth    = "Desonide is used to treat the redness,"
    assert o_qa_result._answer.startswith(s_begin_answer_truth)
    s_end_answer_truth      = "and to sometimes develop red, scaly rashes)."
    assert o_qa_result._answer.endswith(s_end_answer_truth)

    ## Case3 - split a string and count # of words
    assert len(o_qa_result._answer.split())== 53

    ## Case4 - keyword
    assert o_qa_result._keyword=="desonide ointment" 

    print("***Passed excel parsing***")

def test_json_parser():
    o_mdata = InputData("./unittest/data/json/")
    o_mdata.merge_and_split_data(ratio="5:5:0", outdir = "json_test", write_file=True)

    assert len(o_mdata._train_examples)==1
    assert len(o_mdata._valid_examples)==1
    print("===Passed json parsing===")

# call test cases
test_xml_parser()
test_json_parser()
