#!/bin/sh

python3 ./src/test.py

python3 ./src/MedQA.py --eda train:./unittest/test3_json/BioASQ_factoid-6b.json

python3 ./src/MedQA.py --data ./data/dataset/curated_BioASQ_7b/

python3 ./src/MedQA.py --data ./unittest/test4_json/

python3 ./src/MedQA.py --args.end_to_end_tl --data ./unittest/test4_json/
    
