import os

TRAIN_DIR='../../output/models/result_test26_trainset_A_0.3weight_0.8'
DATASET_DIR='/media/disk1/train_test_data/trainset_A/process'

# --batch_size=128    # for TCGA
#  --batch_size=32    # default
#  --weight_decay=0.00004
 # --optimizer=rmsprop \

os.system('python train_image_classifier.py \
  --train_dir=../../output/models/result_test66_trainset_A_0.8_5weight_ex_V3 \
  --dataset_name=tumors \
  --dataset_split_name=train \
  --dataset_dir=/media/disk1/colon-datasets/trainset_A/process/ \
  --model_name=inception_v3 \
  --checkpoint_path=checkpoint/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=100000 \
  --batch_size=64 \
  --learning_rate=0.01 \
  --save_interval_secs=100 \
  --save_summaries_secs=100 \
  --log_every_n_steps=300 \
  --num_epochs_per_decay=1  \
  --optimizer=rmsprop \
  --weight_decay=0.00004 \
  --clone_on_cpu=False')
