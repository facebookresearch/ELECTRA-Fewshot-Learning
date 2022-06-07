#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.
#SBATCH --job-name=sample
#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00

main_dir=$n/space3

type=$1
dataset=$2

FT_method=prompt_tuning
model_name_or_path=$3 # google/electra-large-discriminator 
objective=$4 # dis

output_dir=$main_dir/out/hf/zero_shot_test/$model_name_or_path/${dataset}
mkdir -p $output_dir
lr=1e-5
batch_size=2

max_seq_length=512

if [[ $type == glue ]]; then
python $main_dir/hf/main.py \
  --model_name_or_path $model_name_or_path \
  --task_name $dataset  \
  --do_zero_shot_eval \
  --save_total_limit 2 \
  --max_seq_length $max_seq_length \
  --learning_rate  $lr \
  --num_train_epochs 0 \
  --per_device_train_batch_size $batch_size \
  --per_device_eval_batch_size 32 \
  --output_dir $output_dir \
  --overwrite_output_dir \
  --evaluation_strategy  no \
  --save_strategy steps  \
  --FT_method $FT_method \
  --objective $objective 2>&1 | tee ${output_dir}/log.out
fi

if [[ $type == other ]]; then
python $main_dir/hf/main.py \
  --model_name_or_path $model_name_or_path \
  --dataset_name $dataset  \
  --do_zero_shot_eval \
  --save_total_limit 2 \
  --max_seq_length $max_seq_length \
  --per_device_train_batch_size $batch_size \
  --per_device_eval_batch_size 32 \
  --learning_rate $lr \
  --num_train_epochs 0 \
  --per_device_train_batch_size 4 \
  --output_dir $output_dir \
  --overwrite_output_dir \
  --evaluation_strategy  no \
  --save_strategy steps  \
  --FT_method $FT_method \
  --objective $objective 2>&1 | tee ${output_dir}/log.out
fi

if [[ $type == file ]]; then
declare -A train_files=( ["mr"]=$main_dir/data/downstream/sentiment_dataset/MR/MR.train.tsv ["sst5"]=$main_dir/data/downstream/sentiment_dataset/SST-5/train.txt ["yelp_full"]=/private/home/mengzhouxia/fairseq-py/examples/few_shot/data/yelp_review_full_csv/train.10k.csv ["sst2_aug"]=$out/data/sample_roberta/heuristic/keywords-v2/prompt_dataset/${objective}/${dataset}/train.pt )
declare -A valid_files=( ["mr"]=$main_dir/data/downstream/sentiment_dataset/MR/MR.valid.tsv ["sst5"]=$main_dir/data/downstream/sentiment_dataset/SST-5/dev.txt ["yelp_full"]=/private/home/mengzhouxia/fairseq-py/examples/few_shot/data/yelp_review_full_csv/test.csv ["sst2_aug"]=$out/data/sample_roberta/heuristic/keywords-v2/prompt_dataset/${objective}/${dataset}/valid.pt )
python $main_dir/hf/main.py \
  --model_name_or_path $model_name_or_path \
  --add_task_name $dataset  \
  --train_file ${train_files[$dataset]} \
  --validation_file ${valid_files[$dataset]} \
  --do_zero_shot_eval \
  --save_total_limit 2 \
  --max_seq_length  $max_seq_length \
  --learning_rate $lr \
  --num_train_epochs 0 \
  --per_device_train_batch_size $batch_size \
  --per_device_eval_batch_size 32 \
  --output_dir $output_dir \
  --overwrite_output_dir \
  --evaluation_strategy  no \
  --save_strategy steps  \
  --FT_method $FT_method \
  --objective $objective 2>&1 | tee ${output_dir}/log.out
fi
