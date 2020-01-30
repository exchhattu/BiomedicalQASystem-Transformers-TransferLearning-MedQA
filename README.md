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

## Pipeline:
![alt text](https://github.com/exchhattu/MedQA/blob/master/images/pipeline.png)

### Data and pre-trained model
1. [BioASQ](https://github.com/dmis-lab/bioasq-biobert)
... There are 40K examples for factoid question answer from BioASQ7B. However, unique examples are selected. This resulted 
... 6K examples. Examples such as yes/no and list questions were also excluded.  Explotratory data analysis is carried out to 
... understand the distribution. Interesting, same answer appeared in mutliple contexts. Therefore, answer and its appearance 
... in context is also evaluated. For detail, see a [juypter notebook](https://github.com/exchhattu/MedQA/blob/master/notebook/EDA.ipynb). 
```
$ python3 ./src/EDA.py --eda ./data/dataset/curatedBioASQ/
```

2 [XLnet](https://github.com/zihangdai/xlnet) permutation language model
... Pretrained XLnet model and pytorch_transformer were used for downstream task. 

### Model

#### Unit test
For unit test
```
$ python3 ./src/test.py
```

#### Steps 
To build a model, run following command that performs following tasks:
1. Split the given data into train (90%), valid (5%) and test(5%). 
2. Two parameters - # of epoch and learing rate are optimized using montecarlo sampling 
   and selected the best model. 

#### Usage 
How to run
```
$ python3 ./src/build_model_MedQA.py --end_to_end --data path_to_dir 
```
### Serving 
Best model is used for serving. MXnet is used to top on the pytorch built
model. Model is uploaded in [S3 bucket](https://aws.amazon.com/blogs/machine-learning/deploying-pytorch-inference-with-mxnet-model-server/)
The script used for model serving is at ./serving. To serve model:
```
$ cd ./serving
$ ./automate_serving.sh
```

### Inference 
Go to the (link)[] for inference 

## Future work:
* Train on more biomedical and healthcare and try to reduce balance variance-bias tradeoff.
* Mimic Alexa to find an answer quickly in the document. 

