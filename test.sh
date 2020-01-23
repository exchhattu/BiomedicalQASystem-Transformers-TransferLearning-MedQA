#!/bin/sh

python3 ./src/test.py

python3 ./src/MedQA.py --eda train:./unittest/test3_json/BioASQ_factoid-6b.json

python3 ./src/MedQA.py --data ./data/dataset/curated_BioASQ_7b/

python3 ./src/MedQA.py --data ./unittest/test4_json/BioASQ-train-list-7b-snippet-annotated.json 
