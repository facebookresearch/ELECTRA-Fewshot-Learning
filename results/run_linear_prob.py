# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np
import pdb
def extract_dev(file, keyword):
    try:
        lines = open(file, "r").readlines()
        extracted_lines = []
        for line in lines:
            if line.startswith(keyword):
                extracted_lines.append(line)
        if len(extracted_lines) > 0 and extracted_lines[-1].startswith(keyword):
            num = float(extracted_lines[-1].split(" ")[1])
            return round(num, 4)
        else:
            return None
    except:
        return None


def get_data_type(task):
    if task in ["sst2", "qnli", "rte", "mnli"]:
        return "glue"
    elif task in ["sst5", "mr"]:
        return "file"
    else:
        return "other"


def run_ex(task, model, bs, lr, seed):
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"sbatch {'--exclude=node718' if base_main_dir.startswith('/n') else ''} --job-name={task}_bs{bs}_lr{lr}_{model} -o {output_dir}/slurm_out.log -A pnlp -t 6:00:00"
    ins = f"{prefix} {base_main_dir}/hf/scripts/linear_prob.sh {task} {model} {bs} {lr} {seed}"
    # print(ins)
    # os.system(ins)

base_main_dir = "/n/fs/nlp-mengzhou/space3"
base_dir = f"{base_main_dir}/out/hf"

count = {}

def get_model_name(m):
    if m == "bert":
        return "bert-base-uncased"
    elif m == "bert_large":
        return "bert-large-uncased"
    elif m == "roberta":
        return "roberta-base"
    elif m == "roberta_large":
        return "roberta-large"
    elif m == "electra":
        return "google/electra-base-discriminator"
    else:
        return "google/electra-large-discriminator"

for task in ["sst2", "trec", "ag_news", "imdb"]:
    print(task)
    count[task] = 0
    for model in ["bert"]:
        scores = []
        for seed in [1, 2, 3]:
            collect_scores = []
            for bs in [2, 4, 8]:
                for lr in ["5e-5", "1e-4", "3e-4"]:
                    model_name_or_path = get_model_name(model)
                    output_dir = f"{base_dir}/linear_prob/{task}/{model_name_or_path}/standard_FT/{seed}/bs{bs}_accu4_lr{lr}"
                    output_file = os.path.join(output_dir, "log.out")
                    validation_acc = "no_file"
                    if os.path.exists(output_file):
                        validation_acc = extract_dev(
                            output_file, "validation:")
                    if isinstance(validation_acc, float) and not str(validation_acc).startswith("0"):
                        validation_acc = "wrong"
                        
                    if isinstance(validation_acc, float) and str(validation_acc).startswith("0"):
                        collect_scores.append(validation_acc)

                    if not isinstance(validation_acc, float):
                        count[task] += 1 
                        run_ex(task, model, bs, lr, seed)
            if len(collect_scores) > 0:
                scores.append(max(collect_scores))
        if len(scores) > 0:
            print(f"{round(np.mean(scores) * 100, 2)}\t{round(np.std(scores) * 100, 2)}", end="\t")
        else:
            print("haha\thaha", end="\t")
    print()
print(count)
