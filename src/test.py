'''
@ Insight Data Science
Wed Jan 15 11:38:19 2020

Test cases
'''
import os

from Data import MedicalData 

def test_xml_parser():
    o_mdata = MedicalData()
    # parsing xml file
    o_mdata.parse_xml_input("./unittest/test1/0000001.xml", \
                            is_debug=True)

    icount = o_mdata.parse_xmls("./unittest/test1/", True)
    assert icount == 3

    # parsing excel file
    o_mdata.parse_excel_file("./unittest/test2/test_MedInfo2019-QA-Medications.xlsx", 
                             True)

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

    print("Passed excel parsing.")

def test_json_parser():
    o_mdata = MedicalData()
    num_qas = o_mdata.parse_json_file("./unittest/test3_json/BioASQ_factoid-6b.json", True)
    assert num_qas == 4772
    print("Passed json parsing.")

    o_mdata.write_json_file(os.getcwd(), file_name = "train.json")
    assert os.path.exists(os.path.join(os.getcwd(), "train.json")) == True

# call test cases
test_xml_parser()
test_json_parser()
