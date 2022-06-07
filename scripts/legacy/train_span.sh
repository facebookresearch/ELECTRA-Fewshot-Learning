#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.
#SBATCH --job-name=sample
#SBATCH -A pnlp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=20g


m=$1

main_dir=~/space3
code_dir=$main_dir/hf

task=$2 # copa-retrieval0.15
lr=3e-5
eval_save_steps=5000
objective=$3

if [[ $m == electra ]]; then
  model_name_or_path=google/electra-base-discriminator # roberta-base
fi

if [[ $m == roberta ]]; then
  model_name_or_path=roberta-base
fi

if [[ $m == bert ]]; then
  model_name_or_path=bert-base-uncased
fi

if [[ $task == copa ]]; then
  dataset_config_name=$dataset
  dataset=super_glue
fi

max_train_samples=-1
span_rep_type=$4
FT_method=prompt_span_training
max_seq_length=512
num_train_epochs=1
max_steps=-1
if [[ " $task " =~ copa-v ]]; then 
n_tokens=$3
data_dir=$main_dir/data/created/${task}/prompt_dataset/ntokens${n_tokens}
ex_name=ntokens${n_tokens}_${span_rep_type}_nonsym_eval100
eval_save_steps=100
max_steps=2000
max_train_samples=20000
objective=nonsym_dis
elif [[ " $task " =~ copa-retri ]]; then
data_dir=$main_dir/data/created/copa-retrieval0.15
span_rep_type=$3
ex_name=${span_rep_type}_eval100
num_train_epochs=2
eval_save_steps=100
max_train_samples=20000
fi
output_dir=$main_dir/out/hf/train/$objective/$model_name_or_path/${task}/$ex_name

train_file=$data_dir/train.pt
valid_file=$data_dir/valid.pt
mkdir -p $output_dir

python $code_dir/main.py \
  --model_name_or_path $model_name_or_path \
  --add_task_name ${task}   \
  --do_train \
  --do_eval \
  --save_total_limit 100 \
  --max_seq_length $max_seq_length \
  --learning_rate $lr \
  --num_train_epochs ${num_train_epochs} \
  --eval_steps ${eval_save_steps} \
  --save_steps ${eval_save_steps} \
  --per_device_train_batch_size 8 \
  --output_dir $output_dir \
  --overwrite_output_dir \
  --evaluation_strategy  steps \
  --save_strategy steps  \
  --FT_method $FT_method \
  --objective $objective \
  --train_file $train_file \
  --max_train_samples $max_train_samples \
  --span_rep_type $span_rep_type \
  --validation_file $valid_file \
  --from_nonsym_to_sym_dis 2>&1 | tee ${output_dir}/log.out