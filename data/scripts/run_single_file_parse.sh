#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.


#SBATCH --time 1:00:00
#SBATCH --mem=30g

run_type=$1
index=$2
keyword=$3
v=$4
obj=$5

main_dir=$n/space3/hf
if [[ $run_type == find_keywords ]]; then
  python3 $main_dir/data/keyword_process.py $run_type $index $keyword
fi

if [[ $run_type == find_keywords_aggregate ]]; then
  python3 $main_dir/data/keyword_process.py $run_type None $keyword
fi

if [[ $run_type == aggregate_and_masking ]]; then
  python3 $main_dir/data/keyword_process.py $run_type None None $v $obj
fi

if [[ $run_type == test ]]; then
  python3 $main_dir/data/keyword_process.py $run_type None None $v $obj
fi
