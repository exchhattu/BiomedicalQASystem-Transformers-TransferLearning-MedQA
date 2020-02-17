# MedQA - an automated biomedical question-answer system (Insight Data Science Project)
Artificial intelligence in the biomedical field aims to ease a task. 
One of the time-consuming tasks for biomedical professionals is to read articles. 
This is because there are already 30M articles published and on average 3,000 new 
articles/day are published. Second, it takes a few hours to read an article. 
Due to the incomprehensible amount of data, there is a necessity of an AI-powered 
automated system to speed-up the query from the research articles. MedQA is 
NLP (natural language processing) powered automated question answering system 
that seeks a question and a document as input and outputs a predicted answer within a few seconds. 

## Requirements:
* Python (3.0+)
* NumPy
* Pandas
* PyTorch
* [Transformers](https://github.com/huggingface/transformers)
* [PyTorch-Transformers](https://github.com/rusiaaman/pytorch-transformers) 
* Flask


## Motivation:
* Natural Language Processing - design a tool, automated question answering system 
for biomedical professionals, using AI-powered NLP.
* Transfer Learning - fine-tune generic models trained on the large corpus to achieve 
the specific downstream goal from the biomedical field.
* Model Deployment - serve a model to the users through WebApp.

## Pipeline:
![alt text](https://github.com/exchhattu/MedQA/blob/master/images/pipeline.png)

### Data and pre-trained model
#### Data - [BioASQ](https://github.com/dmis-lab/bioasq-biobert)

There are 40K examples for factoid question answer from BioASQ7B. 
However, unique examples were selected. This resulted in 6K examples. 
Examples such as yes/no and list questions were also excluded. Exploratory data analysis (EDA) 
was carried out to understand the distribution of the length of sequences in questions, 
documents, and answers. Same answers appear at multiple places in different sequences 
with a diverse scenario. 

Detail analysis for EDA can be found in a 
[juypter notebook](https://github.com/exchhattu/MedQA/blob/master/notebook/EDA.ipynb). 
The input for EDA is generated using the following program 
```
$ python3 ./src/EDA.py --eda ./data/dataset/curatedBioASQ/
```

#### Pre-trained Model
Multiple [pre-trained models](https://rajpurkar.github.io/SQuAD-explorer/) are available. 
Few pre-trained models could be selected for fine-tuning but only 
[XLNet](https://github.com/zihangdai/xlnet) was chosen due to the time constraint. 
XLNet was selected among others since it permutes the input sequence to capture the 
dependency of token that otherwise very difficult. [XLnet model](https://arxiv.org/abs/1906.08237) 
and pytorch_transformer were used for the downstream task. 

### Model building 
#### Unit test
```
$ test.sh 
```

#### Summary 
To build a model, run following command that performs following tasks:
1. Randomly suffle split the given data into train (90%), valid (5%) and test(5%). 
2. A
  * Two parameters - # of epoch and learing rate are optimized using montecarlo sampling 
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
