#! /bin/sh

export TASK_NAME=mnli

python3 train_glue_cool.py \
    --task_name $TASK_NAME\
    --model_name_or_path roberta-base\
    --output_dir saved/$TASK_NAME/roberta-cool\
    --num_train_epochs 7\
    --learning_rate 2e-5 \
    --max_length 128 \
    --learning_rate_conv 3e-5\
    --num_outlook_layers 2\
