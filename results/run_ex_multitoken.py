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


def run_ex(task, bs, lr, template_id, span_rep, model, discriminator_head, seed):
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"sbatch {'--exclude=node718' if base_main_dir.startswith('/n') else ''} --job-name={task}_bs{bs}_lr{lr}_{span_rep}_{model} -o {output_dir}/slurm_out.log -A pnlp -t 5:00:00"
    ins = f"{prefix} {base_main_dir}/hf/scripts/run_prompt_span_tuning.sh {task} few_shot {bs} {lr} {template_id} {span_rep} {model} {discriminator_head} {seed}"
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

for task in ["piqa"]:
    print(task)
    count[task] = 0
    for k in [32]:
        for span_rep in ["cls", "mean_prob", "average"]: 
            for model in ["roberta_large", "electra_large"]: # , "roberta_large", "electra_large"]:
                if "roberta" in model and span_rep != "cls":
                    continue
                for template_id in [2]:
                    if span_rep == "cls":
                        discriminator_heads = ["new"]
                    else:
                        discriminator_heads = ["pretrained"]
                    for discriminator_head in discriminator_heads: 
                        scores = []
                        std = []
                        collect_scores = []
                        for seed in [1, 2, 3]:
                            for bs in [2, 4, 8]:
                                for lr in ["1e-5", "2e-5", "3e-5"]:
                                    model_name_or_path = get_model_name(model)
                                    output_dir = f"{base_dir}/few_shot/{task}/{model_name_or_path}/prompt_FT/{span_rep}/head_{discriminator_head}/seed{seed}/bs{bs}_lr{lr}"
                                    output_file = os.path.join(output_dir, "log.out")
                                    validation_acc, fewshot_validation_acc = "no_file", "no_file"
                                    if os.path.exists(output_file):
                                        validation_acc = extract_dev(
                                            output_file, "validation:")
                                        fewshot_validation_acc = extract_dev(
                                                output_file, "fewshot_validation:")
                                    if isinstance(validation_acc, float) and not str(validation_acc).startswith("0"):
                                        validation_acc = "wrong"
                                        fewshot_validation_acc = "wrong"
                                        
                                    if isinstance(validation_acc, float) and str(validation_acc).startswith("0"):
                                        collect_scores.append(fewshot_validation_acc)

                                    if not isinstance(validation_acc, float):
                                        count[task] += 1 
                                        run_ex(task, bs, lr, template_id, span_rep, model, discriminator_head, seed)
                            if len(collect_scores) > 0:
                                scores.append(max(collect_scores))
                                collect_scores = []
                        if len(scores) > 0:
                            print(f"{round(np.mean(scores) * 100, 2)}\t{round(np.std(scores) * 100, 2)}", end="\t")
                        else:
                            print("haha\thaha", end="\t")
        print()
print(count)
