#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.
#SBATCH --job-name=sample
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=50g

main_dir=~/space3

type=$1
dataset=$2
output_dir=$3
objective=$4

output_task_name=$dataset
FT_method=prompt_tuning
model_name_or_path=$output_dir
span_rep_type=average

dataset_config_name=None

if [[ $dataset == copa ]]; then
  dataset_config_name=$dataset
  dataset=super_glue
  FT_method=prompt_span_tuning
elif [[ $dataset == storycloze ]]; then
  FT_method=prompt_span_tuning
elif [[ $dataset == hellaswag ]]; then
  FT_method=prompt_span_tuning
fi

lr=1e-5
batch_size=2
max_seq_length=512

if [[ $type == glue ]]; then
python $main_dir/hf/main.py \
  --model_name_or_path $model_name_or_path \
  --task_name $dataset  \
  --do_eval \
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
  --objective $objective \
  --span_rep_type $span_rep_type \
  --k_shot -1 2>&1 | tee ${output_dir}/eval_${output_task_name}.log
fi

if [[ $type == other ]]; then
python $main_dir/hf/main.py \
  --model_name_or_path $model_name_or_path \
  --dataset_name $dataset  \
  --dataset_config_name $dataset_config_name \
  --do_eval \
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
  --objective $objective \
  --span_rep_type $span_rep_type \
  --k_shot -1 2>&1 | tee ${output_dir}/eval_${output_task_name}.log
fi

if [[ $type == file ]]; then
declare -A train_files=( ["mr"]=$main_dir/data/downstream/sentiment_dataset/MR/MR.train.tsv ["sst5"]=~/fairseq-py/examples/few_shot/data/sentiment_dataset/SST-5/train.txt ["yelp_full"]=/private/home/mengzhouxia/fairseq-py/examples/few_shot/data/yelp_review_full_csv/train.10k.csv ["sst2_aug"]=$out/data/sample_roberta/heuristic/keywords-v2/prompt_dataset/${objective}/${task}/train.pt ["copa-v2"]=$n/space3/data/created/copa-v2/prompt_dataset/ntokens2/train.pt ["agnews_capital_aug"]=$n/space3/meta/data/created/heuristic/keywords-v2/prompt_dataset/$objective/agnews_capital_aug/train.pt ["storycloze"]=/n/fs/nlp-mengzhou/space3/meta/data/downstream/storycloze/spring2016.val.tsv)
declare -A valid_files=( ["mr"]=$main_dir/data/downstream/sentiment_dataset/MR/MR.valid.tsv ["sst5"]=~/fairseq-py/examples/few_shot/data/sentiment_dataset/SST-5/dev.txt ["yelp_full"]=/private/home/mengzhouxia/fairseq-py/examples/few_shot/data/yelp_review_full_csv/test.csv ["sst2_aug"]=$out/data/sample_roberta/heuristic/keywords-v2/prompt_dataset/${objective}/${task}/valid.pt ["copa-v2"]=$n/space3/data/created/copa-v2/prompt_dataset/ntokens2/valid.pt  ["agnews_capital_aug"]=$n/space3/meta/data/created/heuristic/keywords-v2/prompt_dataset/$objective/agnews_capital_aug/valid.pt ["storycloze"]=/n/fs/nlp-mengzhou/space3/meta/data/downstream/storycloze/spring2016.test.tsv)

python $main_dir/hf/main.py \
  --model_name_or_path $model_name_or_path \
  --add_task_name $dataset  \
  --train_file ${train_files[$dataset]} \
  --validation_file ${valid_files[$dataset]} \
  --do_eval \
  --save_total_limit 2 \
  --max_seq_length  $max_seq_length \
  --learning_rate $lr \
  --num_train_epochs 0 \
  --per_device_train_batch_size $batch_size \
  --per_device_eval_batch_size 32 \
  --output_dir $output_dir \
  --overwrite_output_dir \
  --evaluation_strategy no \
  --save_strategy steps  \
  --FT_method $FT_method \
  --objective $objective \
  --span_rep_type $span_rep_type \
  --k_shot -1 2>&1 | tee ${output_dir}/eval_${output_task_name}.log
fi
