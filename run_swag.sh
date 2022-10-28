#! /bin/sh

python3 train_swag_conv.py\
		--train_file data/SWAG/train.csv \
		--predict_file data/SWAG/val.csv \
    --model_name_or_path bert-base-cased\
    --event_record_dir runs/swag_cool\
		--output_dir ~/saved/swag/bert-cool \
		--do_train \
		--do_eval \
		--num_train_epochs 7 \
		--evaluate_during_training \
		--learning_rate 2e-5 \
