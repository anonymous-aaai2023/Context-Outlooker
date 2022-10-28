# Context-Outlooker

This repo provides the anonymous uploading codes for the submisstion.
We run the experiments for eleven NLP tasks, including
- extractive question answering on SQuAD
- multiple choice question answering on SWAG
- question generation on SQuAD
- Malay question answering on Malay SQuAD
- natural language inference on GLUE
- senmatic analysis on IMDB

## Dependency
Main:
- python==3.7
- torch==1.9.0
- transformers=4.6.1
- fairseq==v0.9.0 

Others please refer to **requirements.txt**

## Tasks

### Extractive Question Answering

#### Data Preparation

Download the dataset and put them in ```./data/``` or ```./data_malay/```.
- English SQuAD: [link](https://rajpurkar.github.io/SQuAD-explorer/)
- Malay SQuAD by [malaya](https://malaya.readthedocs.io/en/latest/index.html): [link](https://github.com/huseinzol05/malay-dataset/tree/master/question-answer/squad)

#### Pretrain Model
We use the following pretrained models:
- Bert: [link](https://huggingface.co/bert-base-cased)
- Albert: [link](https://huggingface.co/albert-base-v2?text=The+goal+of+life+is+%5BMASK%5D.)
- Roberta: [link](https://huggingface.co/roberta-base?text=The+goal+of+life+is+%3Cmask%3E.)
- Malay Bert: [link](https://huggingface.co/malay-huggingface/bert-base-bahasa-cased)

#### Model training
Run ```run_qa_en.sh``` or `run_qa_ms.sh` for model training.
```
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
```

### Question Generation

We leverage [ProphnetNet](https://github.com/microsoft/ProphetNet) as the backbone, and insert the proposed module to the encoder.

Go to the directory of ```./ProphetNet/``` for details.

### Multiple Choice Question Answering

#### Data Preparation

Download the [SWAG](https://www.kaggle.com/datasets/jeromeblanchet/swag-nlp-dataset) dataset.

#### Model Training

Run `bash run_swag.sh` for model training

### Natural Language Inference

#### Data Preparation

We use [tensorflow dataset](https://www.tensorflow.org/datasets/catalog/glue)

#### Model Training

Run `bash run_glue.sh` for model training

#### Senmatic AnALYSIS

#### Model Training

Run `bash run_imdb.sh` for model training

### NER or POS Tagging

#### Model Training

Run `bash run_ner.sh` for model training


### P.S. We will further organize all the codes. Appreciate for your consideration.
