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
* [BioASQ](https://github.com/dmis-lab/bioasq-biobert)

There are 40K examples for factoid question answer from BioASQ7B. However, unique examples are selected. This resulted 
6K examples. Examples such as yes/no and list questions were also excluded.  Explotratory data analysis is carried out to 
understand the distribution. Interesting, same answer appeared in mutliple contexts. Therefore, answer and its appearance 
in context is also evaluated. 

For detail, see a [juypter notebook](https://github.com/exchhattu/MedQA/blob/master/notebook/EDA.ipynb). 
```
$ python3 ./src/EDA.py --eda ./data/dataset/curatedBioASQ/
```

* [XLnet](https://github.com/zihangdai/xlnet) permutation language model

Pretrained [XLnet model](https://arxiv.org/abs/1906.08237) and pytorch_transformer were used for downstream task. 

### Model building 
* Unit test
For unit test
```
$ python3 ./src/test.py
```

* Overview 
To build a model, run following command that performs following tasks:
1. Split the given data into train (90%), valid (5%) and test(5%). 
2. Two parameters - # of epoch and learing rate are optimized using montecarlo sampling 
   and selected the best model. 

* Usage 
How to run
```
$ python3 ./src/build_model_MedQA.py --end_to_end --data path_to_dir 
```
### Model management and Serving 
Best model is used for serving. MXnet is used to top on the pytorch built
model. Model is uploaded in [S3 bucket](https://aws.amazon.com/blogs/machine-learning/deploying-pytorch-inference-with-mxnet-model-server/)
The script used for model serving is at ./serving. To serve model:
```
$ cd ./serving
$ ./automate_serving.sh
```

* Results

Five experiments were carried randomly to avoid biases on data speration and 
to build robustness model. Furthermore, Monte carlo sampling was used for each experiment 
to optimize learning rate and epoch. Best model with low loss was kept. This model is
evaluated with test data and here is summary of result. 

* Validation

| Score | Best  | Average | Std.| 
| ------------- |:-------------:| 
| F1    | 49.00 | 100.00  | 1   |
| Exact | 46.00 | 200.00  | 2   |

* Blind Test


### Inference 
Go to the (link)[] for inference 

## Challenges:
* More data - Train on more biomedical and healthcare data to reduce 
  variance-bias tradeoff.
* Context dependency - This also requies large amount of data with multiple
  contet 

## Futures:
* Healthcare chatbots 
* Virtual assistance for healthcare companies
