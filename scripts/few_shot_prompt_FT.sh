#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.
#SBATCH --job-name=sample
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -A pnlp

main_dir=$n/space3

type=$1
dataset=$2
m=$3
batch_size=$4
lr=$5
k_shot=$6
seed=$7
template_id=$8


if [[ $m == electra ]]; then
  model_name_or_path=google/electra-base-discriminator # roberta-base
  objective=dis
elif [[ $m == roberta ]]; then
  model_name_or_path=roberta-base
  objective=mlm
elif [[ $m == bert ]]; then
  model_name_or_path=bert-base-uncased
  objective=mlm
elif [[ $m == bert_large ]]; then
  model_name_or_path=bert-large-uncased
  objective=mlm
elif [[ $m == roberta_large ]]; then
  model_name_or_path=roberta-large
  objective=mlm
elif [[ $m == electra_large ]]; then
  model_name_or_path=google/electra-large-discriminator
  objective=dis
else
  model_name_or_path=$m
  output_dir=$m/FT/${dataset}/bs${batch_size}_lr${lr}
fi
output_dir=$main_dir/out/hf/few_shot/$dataset/$model_name_or_path/shot${k_shot}/prompt_FT/template${template_id}/${seed}/bs${batch_size}_lr${lr}

mkdir -p $output_dir
ls -d $output_dir
accu=1
if [[ $m == *large* ]]; then
if [[ $batch_size == 8 ]]; then batch_size=2; accu=4
else
batch_size=$((batch_size/2))
accu=2
fi
fi

max_seq_length=512

if [[ $type == glue ]]; then
python $main_dir/hf/main.py \
  --model_name_or_path $model_name_or_path \
  --task_name $dataset  \
  --add_task_name $dataset \
  --do_train \
  --do_eval \
  --save_total_limit 1 \
  --max_seq_length $max_seq_length \
  --learning_rate $lr \
  --num_train_epochs 20 \
  --max_steps 1000 \
  --eval_steps 100 \
  --save_steps 2000 \
  --per_device_train_batch_size $batch_size \
  --output_dir $output_dir \
  --overwrite_output_dir \
  --evaluation_strategy  steps \
  --save_strategy steps  \
  --FT_method prompt_tuning \
  --objective $objective \
  --k_shot ${k_shot} \
  --gradient_accumulation_steps $accu \
  --template_id $template_id \
  --data_seed ${seed} 2>&1 | tee ${output_dir}/log.out
fi

if [[ $type == other ]]; then
python $main_dir/hf/main.py \
  --model_name_or_path $model_name_or_path \
  --dataset_name $dataset  \
  --add_task_name $dataset \
  --do_train \
  --do_eval \
  --save_total_limit 1 \
    --save_steps 2000 \
  --max_seq_length $max_seq_length \
  --learning_rate $lr \
  --num_train_epochs 20 \
  --max_steps 1000 \
  --eval_steps 100 \
  --per_device_train_batch_size $batch_size \
  --output_dir $output_dir \
  --overwrite_output_dir \
  --evaluation_strategy  steps \
  --save_strategy steps  \
  --FT_method prompt_tuning \
  --objective $objective \
  --k_shot ${k_shot} \
  --gradient_accumulation_steps $accu \
  --template_id $template_id  \
  --data_seed ${seed}  2>&1 | tee ${output_dir}/log.out
fi

if [[ $type == file ]]; then
declare -A train_files=( ["mr"]=${main_dir}/data/downstream/sentiment_dataset/MR/MR.train.tsv ["sst5"]=${main_dir}/data/downstream/sentiment_dataset/SST-5/train.txt ["yelp_full"]=$n/space3/meta/data/downstream/yelp_review_full_csv/train.10k.csv )
declare -A valid_files=( ["mr"]=${main_dir}/data/downstream/sentiment_dataset/MR/MR.valid.tsv ["sst5"]=${main_dir}/data/downstream/sentiment_dataset/SST-5/dev.txt ["yelp_full"]=$n/space3/meta/data/downstream/yelp_review_full_csv/test.csv )

python $main_dir/hf/main.py \
  --model_name_or_path $model_name_or_path \
  --add_task_name $dataset \
  --train_file ${train_files[$dataset]} \
  --validation_file ${valid_files[$dataset]} \
  --do_train \
  --do_eval \
  --save_total_limit 1 \
  --max_seq_length $max_seq_length \
  --learning_rate $lr \
  --num_train_epochs 20 \
  --max_steps 1000 \
    --save_steps 2000 \
  --eval_steps 100 \
  --per_device_train_batch_size $batch_size \
  --output_dir $output_dir \
  --overwrite_output_dir \
  --evaluation_strategy  steps \
  --save_strategy steps  \
  --FT_method prompt_tuning \
  --objective $objective \
  --k_shot ${k_shot} \
  --gradient_accumulation_steps $accu \
  --template_id $template_id \
  --data_seed ${seed}  2>&1 | tee ${output_dir}/log.out
fi
