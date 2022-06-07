#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00

m=$1
task=$2

main_dir=$n/space3
code_dir=$main_dir/hf

lr=3e-5
eval_save_steps=1000

if [[ $m == electra ]]; then
  model_name_or_path=google/electra-base-discriminator # roberta-base
  objective=dis
fi

if [[ $m == roberta ]]; then
  model_name_or_path=roberta-base
  objective=mlm
fi

if [[ $m == bert ]]; then
  model_name_or_path=bert-base-uncased
  objective=mlm
fi

output_dir=$main_dir/out/hf/train/$objective/$model_name_or_path/${task}/train21k
max_train_samples=-1
FT_method=prompt_training


data_dir=$main_dir/meta/data/created/heuristic/keywords-v2/prompt_dataset/${objective}/${task}
train_file=$data_dir/train.pt
valid_file=$data_dir/valid.pt


max_seq_length=512
mkdir -p $output_dir

python $code_dir/main.py \
  --model_name_or_path $model_name_or_path \
  --add_task_name ${task}   \
  --do_train \
  --do_eval \
  --save_total_limit 100 \
  --max_seq_length $max_seq_length \
  --learning_rate $lr \
  --num_train_epochs 2 \
  --eval_steps ${eval_save_steps} \
  --save_steps ${eval_save_steps} \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 32 \
  --output_dir $output_dir \
  --overwrite_output_dir \
  --evaluation_strategy  steps \
  --save_strategy steps  \
  --FT_method $FT_method \
  --objective $objective \
  --train_file $train_file \
  --max_train_samples $max_train_samples \
  --validation_file $valid_file 2>&1 | tee ${output_dir}/log.out
