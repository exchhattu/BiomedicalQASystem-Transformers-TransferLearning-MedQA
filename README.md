# MedQA - an automated medical question answer system
Aritifical intelligence in health create aims to ease a task easy and fast. In
addition, it can also provide integrated platform to connect professionals who
are in the system such as doctors, patients, scientists, and others. One
of tasks could be to design the automated system in which doctor can find
relevant records in patient note quickly. Similar technology also can be used by other
medical professional such as researchers, clinicians, and pharamacists to get
an answer quickly. 

## Requirements:
* Python version 3.0+
* NumPy
* Pandas
* PyTorch
* [Transformers](https://github.com/huggingface/transformers)
* [PyTorch-Transformers](https://github.com/rusiaaman/pytorch-transformers) 


## Motivation:
* Design a platfrom, automated question answering system for health care domain.
* Implement transfer learning to understand how generic pretrained model can be
  use in different domain. 
* Fine-tune generic models trained on large corpus for specific downstream
  goal. 

## Data and pre-trained model
* [BioASQ](https://github.com/dmis-lab/bioasq-biobert)
* [XLnet](https://github.com/zihangdai/xlnet) permutation language model
* Explotratory Data Analysis
Run follow command to get token level counts
```
$ python3 ./src/EDA.py --eda ./data/dataset/curatedBioASQ/
```
mean  61.570101  15.880612  11.861345  4.574230  31.958541
std  22.813159  6.274397  13.352894  3.620486  28.110541

 


## Model

### Unit test
Unit test for basic functionality 
```
$ python3 ./src/test.py
```

### Steps 
* To build a model, run following command that performs following tasks:
1. Split the given data into train, valid and test in the ratio of 90%, 5%, and 5% 
2. Introduce montecarlo sampling to train the models with different epochs and
   learning rate. Select the best model for downstream task.

Help, 
```
$ python3 ./src/build_model_MedQA.py or ./src/build_model_MedQA.py -h
```

### Usage 
```
$ python3 ./src/build_model_MedQA.py --end_to_end --data path_to_dir 
```


## Result

## Conclusion
