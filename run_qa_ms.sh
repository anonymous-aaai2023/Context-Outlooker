#! /bin/sh

python3 train_squad_ms_cool.py.py --model_type bert \
        --model_name_or_path malay-huggingface/bert-base-bahasa-cased \
        --data_dir ./data_malay/ \
        --output_dir ~/saved/malay/bert-base\
        --tensorboard_save_path ./runs/malay-bert-base\
        --train_file ms-train-2.0.json \
        --predict_file ms-dev-2.0.json \
        --do_train \
        --do_eval \
        --num_train_epochs 7\
        --evaluate_during_training \
        --learning_rate 2e-5 \
        --per_gpu_train_batch_size 24\
        --per_gpu_eval_batch_size 24\
        --version_2_with_negative \
        --max_seq_length 384
