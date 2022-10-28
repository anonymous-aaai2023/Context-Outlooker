#! /bin/sh

python train_ner_cool.py \
    --model_name_or_path roberta-base \
    --dataset_name conll2003 \
    --task_name ner \
    --max_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --output_dir saved/ner/roberta/ \
    --gpu 0\
