#!/bin/sh

MODEL_PATH=/Users/rojan/Kathmandu/CodeSpace/Github/MedicalChat/serving/bestmodels/BioASQ_Jan292020/outdir/

if [ -f .archive ]; do rm -rf ./achive; done
mkdir ./archive

# model archive
model-archiver -f --model-name pytorch_model \
                --model-path $MODEL_PATH 
                --handler pytorch_service:handle 
                --export-path ./archive/


