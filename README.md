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
#### Unit test
```
$ test.sh 
```

#### Overview 
To build a model, run following command that performs following tasks:
1. Randomly split the given data into train (90%), valid (5%) and test(5%). 
2. Two parameters - # of epoch and learing rate are optimized using montecarlo sampling 
   and selected the best model. 

#### Usage 
```
$ ./MedQA.sh 
```

#### Results

Five experiments were carried randomly to avoid biases on data speration and 
to build a robust model. Furthermore, Monte carlo sampling was used for each experiment 
to optimize learning rate and number of epoch. Best model with low loss was kept. This model is
evaluated with test data and here is summary of result. 

* Validation

| Score | Best  | Average | Std. | 
| ------|------ |:-------:|------| 
| Exact | 51.58 |    47.82| 2.67 |
| F1    | 53.32 |    50.46| 2.19 |

* Blind Test

| Score | Best  | Average | Std. | 
| ------|-------|:-------:|-----:| 
| Exact | 51.03 |    46.00| 4.39 |
| F1    | 53.17 |    48.84| 3.81 |

### Model management and Serving 
Best model is used for serving. [Flask](https://www.palletsprojects.com/p/flask/), 
[Nginx](https://www.nginx.com), [gunicorn](https://gunicorn.org), and [AWS](https://aws.amazon.com) 
are used for serving. Server setup is carried out as suggested in a 
[link](https://www.e-tinkers.com/2018/08/how-to-properly-host-flask-application-with-nginx-and-guincorn/).
After completion of server setup, run the following command to start the
server.  

Server side 

```
$ cd PATH_TO_WORK_DIR_IN_SERVER # change accordingly 
$ ./start_server.sh 
```

Client side
If server runs in AWS, client can request a job 
* Open browser (tested on Safari and Chrome)
* Copy http://54.87.194.39 into browser 

## Challenges:
* Data - More biomedical and healthcare data are required to train to reduce 
  variance-bias tradeoff
* Context dependency - This also requies large amount of data with multiple
  context to identify the answers in differnt senario. 

## Futures:
* Healthcare chatbots 
* Virtual assistance for healthcare companies
