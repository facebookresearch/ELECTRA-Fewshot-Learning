#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.
main_dir=$n/space3
script_dir=$main_dir/hf/scripts
declare -A task_type=(["sst2"]=glue ["qnli"]=glue ["rte"]=glue ["mnli"]=glue ["mr"]=file ["sst5"]=file ["ag_news"]=other ["boolq"]=other ["imdb"]=other ["snli"]=other)
for task in sst2; do
  data_type=${task_type[$task]}
  for train_type in prompt_FT; do
    for m in bert roberta electra; do
      if [[ $m == electra ]]; then
        model_name_or_path=google/electra-base-discriminator # roberta-base
      elif [[ $m == roberta ]]; then
        model_name_or_path=roberta-base
      elif [[ $m == bert ]]; then
        model_name_or_path=bert-base-uncased
      fi
      for k in 16 32 256 1024; do
        for seed in 1 3 5 7 9; do
          for bs in 2 4 8; do
            for lr in 1e-5 2e-5 3e-5; do
              output_dir=$main_dir/out/hf/few_shot/$task/$model_name_or_path/shot${k}/${train_type}/${seed}/bs${bs}_lr${lr}
              mkdir -p $output_dir
              if [[ $train_type == standard_FT ]]; then
                # bash $script_dir/run.sh $data_type $task $m $bs $lr $k $seed
                sbatch --job-name ${task}_${k}_${train_type}_m${m}_bs${bs}_lr${lr}_seed${seed} -o $output_dir/slurm.log $script_dir/run.sh $data_type $task $m $bs $lr $k $seed
              elif [[ $train_type == prompt_FT ]]; then
                # bash $script_dir/run_prompt_tuning.sh $data_type $task $m $bs $lr $k $seed
                sbatch --job-name ${task}_${k}_${train_type}_m${m}_bs${bs}_lr${lr}_seed${seed} -o $output_dir/slurm.log $script_dir/run_prompt_tuning.sh $data_type $task $m $bs $lr $k $seed
              fi
            done
          done
        done
      done
    done
  done
done

# v3 training
main_dir=$n/space3
script_dir=$main_dir/hf/scripts
declare -A task_type=(["sst2"]=glue ["qnli"]=glue ["rte"]=glue ["mnli"]=glue ["mr"]=file ["sst5"]=file ["ag_news"]=other ["boolq"]=other ["imdb"]=other ["snli"]=other)
for task in sst2; do
  data_type=${task_type[$task]}
  for train_type in prompt_FT; do
    for m in electra; do
      if [[ $m == electra ]]; then
        model_name_or_path=google/electra-base-discriminator # roberta-base
      elif [[ $m == roberta ]]; then
        model_name_or_path=roberta-base
      elif [[ $m == bert ]]; then
        model_name_or_path=bert-base-uncased
      fi
      for k in 16 32 256 1024; do
        for seed in 1 3 5 7 9; do
          for bs in 2 4 8; do
            for lr in 1e-5 2e-5 3e-5; do
              output_dir=$main_dir/out/hf/few_shot/$task/$model_name_or_path/parallel_dis/hot${k}/${train_type}/${seed}/bs${bs}_lr${lr}
              mkdir -p $output_dir
                # bash $script_dir/run_prompt_tuning.sh $data_type $task $m $bs $lr $k $seed
                sbatch --job-name ${task}_${k}_${train_type}_m${m}_bs${bs}_lr${lr}_seed${seed} -o $output_dir/slurm.log $script_dir/run_prompt_tuning_v2.sh $data_type $task $m $bs $lr $k $seed
            done
          done
        done
      done
    done
  done
done

main_dir=$n/space3
script_dir=$main_dir/hf/scripts
declare -A task_type=(["sst2"]=glue ["qnli"]=glue ["rte"]=glue ["mnli"]=glue ["mr"]=file ["sst5"]=file ["ag_news"]=other ["boolq"]=other ["imdb"]=other ["snli"]=other)
for task in sst2 ag_news; do
  data_type=${task_type[$task]}
  for train_type in prompt_FT; do
    for m in electra; do
      if [[ $m == electra ]]; then
        model_name_or_path=google/electra-base-discriminator # roberta-base
      elif [[ $m == roberta ]]; then
        model_name_or_path=roberta-base
      elif [[ $m == bert ]]; then
        model_name_or_path=bert-base-uncased
      fi
      for k in 16 32 256 1024; do
        for seed in 1 3 5 7 9; do
          for bs in 2 4 8; do
            for lr in 1e-5 2e-5 3e-5; do
              output_dir=$main_dir/out/hf/few_shot/$task/$model_name_or_path/dis-v3/hot${k}/${train_type}/${seed}/bs${bs}_lr${lr}
              mkdir -p $output_dir
                # bash $script_dir/run_prompt_tuning.sh $data_type $task $m $bs $lr $k $seed
                sbatch --job-name ${task}_${k}_${train_type}_m${m}_bs${bs}_lr${lr}_seed${seed} -o $output_dir/slurm.log $script_dir/run_prompt_tuning.sh $data_type $task $m $bs $lr $k $seed
            done
          done
        done
      done
    done
  done
done

## analyze
declare -A task_type=(["sst2"]=glue ["qnli"]=glue ["rte"]=glue ["mnli"]=glue ["mr"]=file ["sst5"]=file ["ag_news"]=other ["boolq"]=other ["imdb"]=other ["snli"]=other)
for model in bert roberta electra; do
  objective=mlm
  if [[ $model == electra ]]; then objective=dis; fi
  for task in rte imdb mr; do 
    data_type=${task_type[$task]}
    bash analyze_v1.sh ${data_type} $task $model $objective
  done
done

model_dir=("/n/fs/nlp-mengzhou/space3/out/hf/few_shot/boolq/bert-base-uncased/shot16/prompt_FT/3/bs2_lr2e-5"
           "/n/fs/nlp-mengzhou/space3/out/hf/few_shot/boolq/roberta-base/shot16/prompt_FT/3/bs2_lr2e-5"
            "/n/fs/nlp-mengzhou/space3/out/hf/few_shot/boolq/google/electra-base-discriminator/shot16/prompt_FT/3/bs2_lr2e-5")
for model in ${model_dir[@]}; do
  objective=mlm
  if [[ $model == *electra* ]]; then objective=dis; fi
  bash analyze_v1.sh other boolq $model $objective 
done

# full training
# train full
main_dir=$n/space3
script_dir=$main_dir/hf/scripts
declare -A task_type=(["sst2"]=glue ["qnli"]=glue ["rte"]=glue ["mnli"]=glue ["mr"]=file ["sst5"]=file ["ag_news"]=other ["boolq"]=other ["imdb"]=other ["snli"]=other)
for task in boolq; do
data_type=${task_type[$task]}
for m in bert_large electra_large; do
  for batch_size in 4; do
    for lr in 1e-5 2e-5 3e-5; do
    if [[ $m == electra_large ]]; then
      model_name_or_path=google/electra-large-discriminator # roberta-base
    elif [[ $m == roberta ]]; then
      model_name_or_path=roberta-base
    elif [[ $m == bert_large ]]; then
      model_name_or_path=bert-large-uncased
    elif [[ $m == roberta_large ]]; then
      model_name_or_path=roberta-large
    fi
    output_dir=$main_dir/out/hf/FT/$task/$model_name_or_path/bs${batch_size}_accu4_lr${lr} 
    mkdir -p $output_dir
    # echo $output_dir
    # bash $script_dir/FT.sh $data_type $task $m $bs $lr 
    sbatch --job-name ${task}_full_FT_m${m}_bs${batch_size}_lr${lr} -o $output_dir/slurm.log $script_dir/FT.sh $data_type $task $m $batch_size $lr 
    done
  done
done
done

for task in sst5 ag_news boolq imdb snli; do
  scancel $(sqkl | grep full | cut -d" " -f1)
done