#!/usr/bin/env bash
set -e

PRETRAINED_CHECKPOINT_DIR=checkpoint #The directory where checkpoints are located or pretrained model path
TRAIN_DIR=../../output/models/result_test_trainset    #model path
DATASET_DIR=/media/disk1/colon-datasets/trainset/process/       #training set path

python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=tumors \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=100000 \              #max steps, [50000,100000, 150000]
  --batch_size=64 \
  --learning_rate=0.01 \
  --save_interval_secs=100 \
  --save_summaries_secs=100 \
  --log_every_n_steps=300 \
  --num_epochs_per_decay=1  \
  --optimizer=rmsprop \
  --weight_decay=0.00004 \
  --clone_on_cpu=False
