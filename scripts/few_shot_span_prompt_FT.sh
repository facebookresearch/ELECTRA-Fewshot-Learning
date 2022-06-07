#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.
#SBATCH --job-name=sample
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

declare -A task_type=(["copa"]="super_glue" ["storycloze"]="file" ["piqa"]="other" ["hellaswag"]="other")

dataset=$1
type=$2 # when use file, few-shot setting, when use other, full shot setting
batch_size=$3
lr=$4
template_id=$5
span_rep_type=$6
model=$7
discriminator_head=$8 # new/pretrained for cls reps
data_seed=$9

if [[ $dataset == copa ]]; then
dataset_config=super_glue
dataset=copa
fi 

k_shot=-1

main_dir=$n/space3
code_dir=$main_dir/hf

eval_save_steps=100
max_steps=1000
objective=dis
if [[ $model == electra ]]; then
model_name_or_path=google/electra-base-discriminator # roberta-base
fi

if [[ $model == electra_large ]]; then
model_name_or_path=google/electra-large-discriminator
fi

if [[ $model == roberta ]]; then
model_name_or_path=roberta-base
fi

if [[ $model == roberta_large ]]; then
model_name_or_path=roberta-large
fi

FT_method=prompt_span_tuning
max_seq_length=256
if [[ $task == hellaswag ]]; then max_seq_length=512; fi

output_dir=$main_dir/out/hf/few_shot/$dataset/$model_name_or_path/prompt_FT/${span_rep_type}/head_${discriminator_head}/seed${data_seed}/bs${batch_size}_lr${lr}
declare -A train_files=( ["storycloze"]=$main_dir/data/downstream/storycloze/withheadline/fewshot_train_${data_seed}.tsv
                         ["piqa"]=$main_dir/data/downstream/piqa/fewshot_train_${data_seed}.jsonl
                         ["copa"]=$main_dir/fewglue/FewGLUE/COPA/fewshot_train_${data_seed}.jsonl
                         ["hellaswag"]=$main_dir/data/downstream/hellaswag/fewshot/fewshot_train_${data_seed}.jsonl)
declare -A valid_files=( ["storycloze"]=$main_dir/data/downstream/storycloze/withheadline/val.tsv
                         ["piqa"]=$main_dir/data/downstream/piqa/fewshot_val.jsonl
                         ["copa"]=$main_dir/fewglue/FewGLUE/COPA/fewshot_val.jsonl
                         ["hellaswag"]=$main_dir/data/downstream/hellaswag/fewshot/fewshot_val.jsonl)
mkdir -p $output_dir
echo $output_dir

train_file=${train_files[$dataset]}
valid_file=${valid_files[$dataset]}
echo $train_file 
echo $valid_file

python $code_dir/main.py \
  --model_name_or_path $model_name_or_path \
  --add_task_name ${dataset}   \
  --do_train \
  --do_eval \
  --save_total_limit 100 \
  --max_seq_length $max_seq_length \
  --max_steps 1000 \
  --eval_steps ${eval_save_steps} \
  --save_steps 2000 \
  --per_device_train_batch_size ${batch_size} \
  --learning_rate $lr \
  --output_dir $output_dir \
  --overwrite_output_dir \
  --evaluation_strategy  steps \
  --save_strategy steps  \
  --FT_method $FT_method \
  --objective $objective \
  --train_file $train_file \
  --span_rep_type $span_rep_type \
  --k_shot $k_shot \
  --data_seed ${data_seed} \
  --template_id ${template_id} \
  --discriminator_head ${discriminator_head} \
  --validation_file $valid_file 2>&1 | tee ${output_dir}/log.out
