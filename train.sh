#!/bin/bash

if [ "$#" -ne 2 ]
  then
    echo "Usage: train.sh DATA_DIR OUTPUT_DIR"
    exit
fi

DATA_DIR=$1
OUTPUT_DIR=$2

pip install -r requirements.txt
python -m spacy download en

python train_bilstm.py --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --augmented \
    --batch_size 50 --gradient_accumulation_steps 1 \
    --lr 1e-3 --lr_schedule warmup --warmup_steps 50 \
    --epochs 1 \
    --do_train