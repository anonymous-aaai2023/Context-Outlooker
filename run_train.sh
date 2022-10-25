#! /bin/sh

python3 train_squad_en_cool.py --model_type bert\
	--model_name_or_path bert-base-cased\
	--data_dir ./data/ \
	--architecture global-to-local\
	--with_conv_block\
	--output_dir ~/saved/squad2.0/bert-base\
	--tensorboard_save_path ./runs/bert-base\
	--train_file train-v2.0.json \
	--predict_file dev-v2.0.json \
	--do_train \
	--do_eval \
	--num_outlook_layers 2\
	--num_train_epochs 7\
	--evaluate_during_training \
	--per_gpu_train_batch_size 24\
	--per_gpu_eval_batch_size 24\
	--version_2_with_negative \
	--max_seq_length 384
