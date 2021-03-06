# MedQA - an automated biomedical question-answer system 
Artificial intelligence in the biomedical field aims to ease a task. 
One of the time-consuming tasks for biomedical professionals is to read articles. 
This is because there are already 30M articles and on average 3,000 new 
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
* Natural Language Processing - design a tool, automated biomedical question answering system 
, using AI-powered NLP.
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
documents, and answers. The same answers appear at multiple places in different sequences are also analyzed. 

Detail analysis for EDA can be found in a 
[juypter notebook](https://github.com/exchhattu/MedQA/blob/master/notebook/EDA.ipynb). 
The input for EDA is generated using the following program. However, before running the program, 
please download the data from the above link and provide the path in --eda
parameter. 
```
$ python3 ./src/EDA.py --eda ./data/dataset/curatedBioASQ/
```

#### Pre-trained Model
Multiple [pre-trained models](https://rajpurkar.github.io/SQuAD-explorer/) 
are publicly available. Few pre-trained models could be selected for 
fine-tuning but only [XLNet](https://github.com/zihangdai/xlnet) was 
chosen due to the time constraint. Although XLNet 
is an autoregressive model as described in a paper, it is able to see 
the token from both directions. In addition, it uses Transformer-XL, 
which includes relative positional encoding and recurrence mechanism 
to gain the long-distance word relation. Therefore, in addition to content 
stream attention equivalent to the self-attention layer, query stream attention 
is also used. [Pytorch implementation](https://huggingface.co/) was used in 
fine-tuning [XLnet](https://arxiv.org/abs/1906.08237) for the downstream task.
Open-source codes were primarily used but modified accordingly. 
All updated codes were in src directory.

### Model building 
#### Unit test
```
$ test.sh 
```

#### Summary 

To build a model, the following steps were performed:
1. Randomly shuffle the indexes of given sequences (not the sequence) and 
   split it into the train (90%), valid (5%) and test(5%) and repeated 
   this experiment five times.
2. Training
    * Two hyperparameters (# of epoch and very-small learning rate) are optimized 
      and selected the best model.
    * Freeze transformer layers and train the last linear layer only.
    * The models are trained at AWS using EC2 instance (p2.xlarge).    

#### Usage 
```
$ ./MedQA.sh 
```

#### Results

Two approaches were tested to fine-tune the pre-trained model. 
For each, multiple experiments were carried out randomly to build a 
robust model. First, the weights of the transformer were unchanged and 
only optimized the weight of the last layer. 
This model performed the best F1 score of 27 and 30 on validation and 
test datasets. The model could not recognize the biological vocabularies 
precisely and showed a lower F1 score. 
Therefore, as a second approach, the weights of the transformer were 
updated with a smaller learning rate to minimize the model being overfitted. 
This model improved the performance of both datasets as shown below in 
the table. This is expected since while updating the weight of the 
pre-trained model with biological knowledge can downgrade the role of 
the least influence features of it.  The model with the best F1 score on 
validation and the independent datasets was selected for model serving.

| Score | validation | Independent | 
| ------|-----------:|-------------| 
| F1    | 53.32      |53.17        |

### Model Management and Serving 
The best model is used for serving. [Flask](https://www.palletsprojects.com/p/flask/), 
[Nginx](https://www.nginx.com), [gunicorn](https://gunicorn.org), and [AWS](https://aws.amazon.com) 
are used for serving. Server setup is carried out as suggested in a 
[link](https://www.e-tinkers.com/2018/08/how-to-properly-host-flask-application-with-nginx-and-guincorn/).
After completion of server setup, run the following command to start the server.  

Server side 
```
$ cd PATH_TO_WORK_DIR_IN_SERVER # change accordingly 
# Copy the content of ./serving/flask
$ ./start_server.sh 
```

Client side
If server runs in AWS, client can request a job 
* Open browser (tested on Safari and Chrome)
* Copy http://54.87.194.39 into browser 

## Challenges:
* More data are required for training to reduce variance-bias tradeoff
* Due to the diversity in tokens dependency of the given answer, the following 
  two cases are necessary to address and for which a large amount of data 
  is necessary.
  1. When a question is slightly changed to seeking the same answer, predictions can be different. 
  2. When the same question is asked in different documents, the answer might be predicted differently. 

## Futures:
* First, automate a method to search in the entire document instead of a 
  small paragraph. Second, it would have a high impact, if the answer can 
  be provided learning from entire articles available in PubMed.  
* Transfer a technique to query the doctor's note. 

## Reference:
* [PubMed](https://www.ncbi.nlm.nih.gov/pubmed/)
