#!/bin/sh

python3 ./src/test.py

python3 ./src/MedQA.py --eda train:./unittest/test3_json/BioASQ_factoid-6b.json


python3 ./src/MedQA.py --data ./unittest/test4_json/

# use pretrained model
python3 ./src/MedQA.py --use_pretrained_model --data ./data/dataset/curated_BioASQ_7b/

python3 ./src/MedQA.py --end_to_end --data ./unittest/test4_json/

python3 ./src/MedQA.py --end_to_end --data ./unittest/test_squad/
    
