"""
Rojan Shresetha, Insight Data Science
Wed Jan 29 09:54:09 2020
"""

import os, io, json, time

import numpy as np
import collections

import torch
from torch.autograd import Variable
# from torchvision import transforms
import torch.nn.functional as F

from torch.utils.data import (DataLoader, 
                                RandomSampler, 
                                SequentialSampler,
                                TensorDataset)

from pytorch_transformers import  XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer
from utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended,
                         get_final_text, _compute_softmax)
from ServeMedQAFlask import MedQAInference

from flask import Flask, jsonify, request, render_template
app = Flask(__name__)

meq_qa_inference = MedQAInference()
path_to_model="./model/"
# url="https://medqa.s3.amazonaws.com/"
# path_to_model = requests.get(url).json()
meq_qa_inference.initialize(path_to_model)


def get_prediction(path_to_file, input_question):
    start_time = time.time()
    examples, features, dataset =  meq_qa_inference.preprocess_data(path_to_file, input_question)
    all_results = meq_qa_inference.inference(dataset, examples, features)
    data = meq_qa_inference.postprocess(examples, features, all_results)
    elapsed_time = time.time() - start_time
    return data, elapsed_time 

@app.route("/", methods=['GET', 'POST'])
def predict():
    if request.method=='POST': 
        print("Coding: hi ", request)
        if 'file' not in request.files:
            return redirect(request.url)
        # file = request.files.get('file')
        file = request.files['file']
        if not file: return
        input = request.form['Question']
        if not input: return
        data, etime = get_prediction(file.stream, input)
        return render_template('result.html', question=input, answer=data, con_time = etime)
    return render_template('index.html')

if __name__=='__main__':
    # app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
    # app.run(debug=True)
