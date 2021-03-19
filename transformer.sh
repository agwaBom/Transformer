#!/usr/bin/env bash

function make_dir () {
    if [[ ! -d "$1" ]]; then
        mkdir $1
    fi
}

DATA_DIR=data
MODEL_DIR=tmp

make_dir $MODEL_DIR

DATASET=python
CODE_EXTENSION=original_subtoken
JAVADOC_EXTENSION=original


function train () {

echo "============TRAINING============"

MODEL_NAME=$1

PYTHONPATH=CUDA_VISIBLE_DEVICES=2 python -W ignore train.py \
--data_workers 5 \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/ \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--train_src train/code.${CODE_EXTENSION} \
--train_tgt train/javadoc.${JAVADOC_EXTENSION} \
--valid_src dev/code.${CODE_EXTENSION} \
--valid_tgt dev/javadoc.${JAVADOC_EXTENSION} \
--uncase True \
--use_src_word True \
--use_src_char False \
--use_tgt_word True \
--use_tgt_char False \
--max_src_len 400 \
--max_tgt_len 30 \
--emsize 512 \
--fix_embeddings False \
--src_vocab_size 50000 \
--tgt_vocab_size 30000 \
--share_decoder_embeddings True \
--max_examples -1 \
--batch_size 32 \
--test_batch_size 64 \
--num_epochs 200 \
--model_type transformer \
--num_head 8 \
--d_k 64 \
--d_v 64 \
--d_ff 2048 \
--src_pos_emb False \
--tgt_pos_emb True \
--max_relative_pos 32 \
--use_neg_dist True \
--nlayers 6 \
--trans_drop 0.2 \
--dropout_emb 0.2 \
--dropout 0.2 \
--copy_attn True \
--early_stop 20 \
--warmup_steps 0 \
--optimizer adam \
--learning_rate 0.0001 \
--lr_decay 0.99 \
--valid_metric bleu \
--checkpoint True

}


function test () {

echo "============TESTING============"

MODEL_NAME=$1

PYTHONPATH=CUDA_VISIBLE_DEVICES=2 python -W ignore train.py \
--only_test True \
--data_workers 5 \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/ \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--valid_src test/code.${CODE_EXTENSION} \
--valid_tgt test/javadoc.${JAVADOC_EXTENSION} \
--uncase True \
--max_src_len 400 \
--max_tgt_len 30 \
--max_examples -1 \
--test_batch_size 64

}

train $1
test $1
