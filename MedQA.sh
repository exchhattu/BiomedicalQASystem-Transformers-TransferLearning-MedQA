#!/bin/sh

SQUAD_DIR=`pwd` 

i=1
while [ $i -le 6 ];
do
   out_dir=outdir_$i
   test_out_dir=test_out_dir_$i

   # MedQA protocol
   # data 
   python3 $SQUAD_DIR/src/MedQA.py \
	--data $SQUAD_DIR/InputData \
   	--outdir split_train_valid

   if [ ! -f $SQUAD_DIR/split_train_valid/train.json ]
   then
      echo "NOT FOUND - " $SQUAD_DIR/split_train_valid/train.json
      exit 
   fi
   
   if [ ! -f $SQUAD_DIR/split_train_valid/valid.json ]
   then
      echo "NOT FOUND - " $SQUAD_DIR/split_train_valid/valid.json 
      exit 
   fi
   
   # train 
   python3 $SQUAD_DIR/src/run_medqa.py \
         --overwrite_cache \
         --overwrite_output_dir \
         --model_type xlnet \
         --model_name_or_path xlnet-base-cased \
         --train_file $SQUAD_DIR/split_train_valid/train.json \
         --predict_file $SQUAD_DIR/split_train_valid/valid.json \
         --do_train \
         --do_eval \
         --do_lower_case \
         --max_seq_length 192 \
         --doc_stride 128 \
         --output_dir $SQUAD_DIR/$out_dir \
         --save_steps 500000 \
         --per_gpu_eval_batch_size=3   \
         --per_gpu_train_batch_size=3   \
         --num_sample 1 
   
   if [ ! -d $SQUAD_DIR/$out_dir ]
   then
      echo "NOT FOUND - " $SQUAD_DIR/$out_dir 
      exit 
   fi

   # independent test run 
   python3 $SQUAD_DIR/src/run_medqa.py \
         --model_type xlnet \
         --model_name_or_path $SQUAD_DIR/$out_dir \
         --predict_file $SQUAD_DIR/split_train_valid/test.json \
         --do_eval \
         --do_lower_case \
         --max_seq_length 192 \
         --doc_stride 128 \
         --output_dir $SQUAD_DIR/$test_out_dir 

   # if split_train_valid exists, rename it to save for later
   if [ -d $SQUAD_DIR/split_train_valid ]
   then
      mv $SQUAD_DIR/split_train_valid $SQUAD_DIR/split_train_valid_$i 
   fi 
  i=$(( i+1 ))
  echo "FINISHED!!!"
done
