#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.
script=$n/space3/hf/data/scripts/run_single_file_parse.sh

for keyword in Yes-v1 Yes-v2 No-v1 No-v2 Maybe-v1 Maybe-v2; do #  which_means which_implies Similarly-v1 Similarly-v2 In_contrast On_the_contrary Yes-v1 Yes-v2 No-v1 No-v2 Maybe-v1 Maybe-v2 as because World world Business business Sports sports Tech tech; do
for i in {00..89} {9000..9101}; do
  log_dir=$n/space3/meta/data/created/heuristic/keywords-v2/keyword_dataset/${keyword}/logs
  mkdir -p $log_dir
  sbatch --job-name keyword_${i} -o $log_dir/${i}.log $script find_keywords $i $keyword
# bash $script find_keywords $i $keyword 2>&1 | tee $log_dir/${i}.log
done
done

# good bad terrible okay great
for keyword in Yes-v1 Yes-v2 No-v1 No-v2 Maybe-v1 Maybe-v2; do # which_means which_implies Similarly-v1 Similarly-v2 In_contrast On_the_contrary Yes-v1 Yes-v2 No-v1 No-v2 Maybe-v1 Maybe-v2 as because World world Business business Sports sports Tech tech; do
log_dir=$n/space3/meta/data/created/heuristic/keywords-v2/keyword_dataset/${keyword}/logs
bash $script find_keywords_aggregate None $keyword 2>&1 | tee $log_dir/all.log
done

template_names=nli_v2
object=mlm
log_dir=$n/space3/meta/data/created/heuristic/keywords-v2/prompt_dataset/${object}/${template_names}_aug
mkdir -p $log_dir/logs
bash $script aggregate_and_masking  None None $template_names $object 2>&1 | tee $log_dir/log.txt


template_names=boolq
object=dis
log_dir=/checkpoint/mengzhouxia/data/sample_roberta/heuristic/keywords-v2/prompt_dataset/${object}/${template_names}_aug
mkdir -p $log_dir/logs
bash $script aggregate_and_masking  None None $template_names $object 2>&1 | tee $log_dir/log.txt

outdir=/n/fs/nlp-mengzhou/space3/data/created/copa-v2/preprocess
mkdir -p $outdir
python $n/space3/hf/data/multitoken.py 2>&1 | tee $outdir/log.txt

n_tokens=1
func=find_replacement
outdir=/n/fs/nlp-mengzhou/space3/data/created/copa-v3/replacement/ntokens${n_tokens}
mkdir -p $outdir
python3 $n/space3/hf/data/multitoken.py $func $n_tokens 2>&1 | tee $outdir/log.txt 

n_tokens=2
func=get_prompt
outdir=/n/fs/nlp-mengzhou/space3/data/created/copa-v3/prompt_dataset/ntokens${n_tokens}
mkdir -p $outdir
python3 $n/space3/hf/data/multitoken.py $func $n_tokens 2>&1 | tee $outdir/log.txt 