#! /bin/sh

fairseq-generate \
	qg/processed \
	--path qg/finetune_qg_checkpoints_both/checkpoint_best.pt \
	--user-dir prophetnet \
	--task translation_prophetnet \
	--batch-size 80 \
	--gen-subset test \
	--beam 1 \
	--num-workers 4 \
	--no-repeat-ngram-size 3 \
	--lenpen 1.5 2>&1 > $OUTPUT_FILE grep ^H output_ck5_pelt1.5_cool_beam1 | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > qg/sort_hypo_ck5_pelt1.5_cool_beam1.txt
