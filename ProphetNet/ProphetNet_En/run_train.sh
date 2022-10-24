#! /bin/sh

fairseq-train \
	--fp16 \
	--batch-size 200\
	--user-dir ./prophetnet-cool-conv --task translation_prophetnet --arch ngram_transformer_prophet_large \
	--reset-optimizer \
	--optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.1 \
	--lr 0.00001 --min-lr 1e-09 \
	--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
	--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
	--criterion ngram_language_loss --label-smoothing 0.1 \
	--update-freq 1  --max-tokens 1400\
	--num-workers 4 \
	--load-from-pretrained-model pretrained_checkpoints/prophetnet_en.pt \
	--ddp-backend=no_c10d --max-epoch 10 \
	--max-source-positions 512 --max-target-positions 512 \
	--skip-invalid-size-inputs-valid-test \
	--save-dir qg/finetune_qg_checkpoints_both \
	--keep-last-epochs 10 \
	--tensorboard-logdir qg/finetune_qg_tensorboard_both \
	qg/processed
