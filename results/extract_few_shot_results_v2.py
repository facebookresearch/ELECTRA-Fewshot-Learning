# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.
""" This version takes data seeds into account """
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


def run_ex(type, task, m, bs, lr, k, seed, template_id):
    data_type = get_data_type(task)
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"sbatch --mem=50g -A pnlp --exclude=node718 -t 11:00:00 --job-name={type}_{task}_bs{bs}_lr{lr}_k{k}_m{m} -o {output_dir}/slurm_out.log"
    if task == "imdb":
        prefix = f"sbatch -A pnlp --exclude=node718 -t 11:00:00 --mem=20g --job-name={type}_{task}_bs{bs}_lr{lr}_k{k}_m{m} -o {output_dir}/slurm_out.log"
    if type == "prompt_FT":
        if tt.endswith("v1") or tt.endswith("v3") or tt.endswith("vlarge"):
            ins = f"{prefix} $n/space3/hf/scripts/run_prompt_tuning.sh {data_type} {task} {m} {bs} {lr} {k} {seed} {template_id}"
        else:
            ins = f"{prefix} $n/space3/hf/scripts/run_prompt_tuning_v2.sh {data_type} {task} {m} {bs} {lr} {k} {seed}" 
    elif type == "standard_FT":
        ins = f"{prefix} $n/space3/hf/scripts/run.sh {data_type} {task} {m} {bs} {lr} {k} {seed}"
    # print(ins)
    # os.system(ins)


def get_model_name(m):
    if m.startswith("bert"):
        if "large" not in m:
            return "bert-base-uncased"
        else:
            return "bert-large-uncased"
    elif m.startswith("roberta"):
        if "large" not in m:
            return "roberta-base"
        else:
            return "roberta-large"
    else:
        if "large" not in m:
            return "google/electra-base-discriminator"
        else:
            return "google/electra-large-discriminator"


base_dir = "/n/fs/nlp-mengzhou/space3/out/hf"


def get_objective(model, type):
    if type == "standard_FT":
        objective = ""
    else:
        if "bert" in model:
            objective = "mlm"
        else:
            objective = "dis"
    return objective


# v1: normal electra v2: contrastive electra
tt = "few_shot_vlarge" 
def get_models(tt):
    if tt.endswith("v1"):
        return ["bert", "roberta", "electra"]
    elif tt.endswith("vlarge"):
        return ["bert_large", "roberta_large", "electra_large"]
    else:
        return ["electra"]

def get_types(tt):
    if tt.endswith("v1"):
        return ["standard_FT", "prompt_FT"]
    else:
        return ["prompt_FT"]
count = {}
# ["sst2", "sst5", "snli", "mnli", "rte", "qnli", "ag_news", "boolq", "mr", "imdb"]:
models = get_models(tt)
types = get_types(tt)

for task in ["snli"]: # ["sst2", "sst5", "snli", "mnli", "rte", "qnli", "ag_news", "boolq", "mr"]:
    print(task)
    count[task] = 0
    for k in [16]:
        if k == 1024 and task in ["sst5", "rte"]:
            continue
        for m in models:
            for type in ["standard_FT", "prompt_FT"]: 
                for template_id in [2]:
                    scores = []
                    for seed in [1, 3, 5, 7, 9]:
                        collect_scores = []
                        for bs in [2, 4, 8]:
                            for lr in ["1e-5", "2e-5", "3e-5"]:
                                model = get_model_name(m)
                                objective = get_objective(model, type)
                                if tt.endswith("v1") or tt.endswith("vlarge"):
                                    output_dir = f"{base_dir}/few_shot/{task}/{model}/shot{k}/{type}/template{template_id}/{seed}/bs{bs}_lr{lr}"
                                    if not os.path.exists(output_dir):
                                        output_dir = f"{base_dir}/few_shot/{task}/{model}/shot{k}/{type}/{seed}/bs{bs}_lr{lr}"
                                elif tt.endswith("v2"):
                                    output_dir = f"{base_dir}/few_shot/{task}/{model}/parallel_dis/shot{k}/{type}/{seed}/bs{bs}_lr{lr}"
                                else:
                                    output_dir = f"{base_dir}/few_shot/{task}/{model}/dis-v3/shot{k}/{type}/{seed}/bs{bs}_lr{lr}"

                                output_file = os.path.join(output_dir, "log.out")
                                validation_acc, fewshot_validation_acc, fewshot_mismatch_validation_acc = "no_file", "no_file", "no_file"
                                if os.path.exists(output_file):
                                    validation_acc = extract_dev(
                                        output_file, "validation:")
                                    if task == "mnli":
                                        fewshot_validation_acc = extract_dev(
                                            output_file, "fewshot_validation_matched:")
                                        fewshot_mismatch_validation_acc = extract_dev(
                                            output_file, "fewshot_validation_mismatched:")
                                    else:
                                        fewshot_validation_acc = extract_dev(
                                            output_file, "fewshot_validation:")
                                if isinstance(validation_acc, float) and not str(validation_acc).startswith("0"):
                                    validation_acc = "wrong"
                                    fewshot_validation_acc = "wrong"
                                
                                if isinstance(validation_acc, float) and str(validation_acc).startswith("0"):
                                    collect_scores.append(fewshot_validation_acc)

                                if not isinstance(validation_acc, float):
                                    # run_ex(type, task, m, bs, lr, k, seed, template_id)
                                    # print(output_dir)
                                    count[task] += 1
                        if len(collect_scores) > 0:
                            scores.append(max(collect_scores))
                    if len(scores) > 0:
                        scores = [s for s in scores if s > min(scores)]
                        print(f"{round(np.mean(scores) * 100, 2)}\t{round(np.std(scores) * 100, 2)}", end="\t")
                    else:
                        print("haha\thaha", end="\t")
            print()
print(count)
