'''
@ Insight Data Science
Wed Jan 15 11:38:19 2020

Test cases
'''

from Data import MedicalData 

def test_xml_parser():
    o_mdata = MedicalData()
    o_mdata.parse_xml_input("./unittest/test1/0000001.xml")


# call test cases
test_xml_parser()
