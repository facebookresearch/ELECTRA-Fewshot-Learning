# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np
import json

dir = "/n/fs/nlp-mengzhou/space3/data/downstream/piqa"
seed = 1


input_file = os.path.join(dir, "train.jsonl")
input_labels = os.path.join(dir, "train-labels.lst")

train_lines = open(input_file, "r").readlines()
train_labels = open(input_labels, "r").readlines()

def sample_data(num):
    index = np.random.permutation(len(train_lines))[:num]
    sampled_lines = [train_lines[i] for i in index]
    sampled_labels = [train_labels[i] for i in index]

    sampled_line_labels = []
    for line, label in zip(sampled_lines, sampled_labels):
        json_line = json.loads(line)
        json_line["label"] = int(label.strip())
        sampled_line_labels.append(json_line)
    return sampled_line_labels

output_file = os.path.join(dir, f"fewshot_train_{seed}.jsonl")
f = open(output_file, "w")
for line in sample_data(32):
    f.write(json.dumps(line) + "\n")
    

output_file = os.path.join(dir, f"fewshot_val_{seed}.jsonl")
f = open(output_file, "w")
for line in sample_data(32):
    f.write(json.dumps(line) + "\n")
    

