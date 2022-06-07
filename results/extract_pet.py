# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np

def extract(log):
    acc = None
    try:
        lines = open(log, "r").readlines()
        for line in lines:
            if "Best" in line:
                index = line.index("Best acc so far: ")
                acc = float(line[index+len("Best acc so far: "):].split(",")[0])
    except:
        return None
    return acc

def extract_hellaswag_file(log):
    acc = float(open(log, "r").readlines()[-1].strip().split(" ")[-3])
    return acc

base_main_dir = "/n/fs/nlp-mengzhou/space3"
def run_ex(output_dir, task, bs, lr, pattern, seed, model):
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"sbatch {'--exclude=node718'} --job-name={task}_bs{bs}_lr{lr}_{pattern}_{model} -o {output_dir}/{model}-slurm.log -A pnlp -t 5:00:00"
    ins = f"{prefix} {base_main_dir}/pet/scripts/run.sh {task} {bs} {lr} {seed} roberta {model} {pattern}"
    # print(output_dir)
    # print(ins)
    # os.system(ins)

def extract_hellaswag(output_dir):
    eval_dir = os.path.join(output_dir, "evaluation")
    
    total = 0
    all = 0
    for i in range(10):
        log = os.path.join(eval_dir, f"eval-log-{i}.txt")

        try:
            acc = extract_hellaswag_file(log)
            if i < 9:
                total += acc * 1000
                all += 1000
            else:
                total +=  acc * 1042
                all += 1042
        except:
            continue
    if all == 0:
        return None
    else:
        return total / all

        
for task in ["copa"]:
    print(task)
    valid_ex_count = 0
    all_ex_count = 0
    for model in ["roberta-base", "roberta-large"]:
        print(model)
        best_acc = []
        for seed in [1, 2, 3]:
            all_acc = []
            for pattern in [0, 1]:
                for bs in [2, 4, 8]:
                    for lr in ["1e-5", "2e-5", "3e-5"]:
                        output_dir = f"/n/fs/nlp-mengzhou/space3/out/pet/{task}/pattern{pattern}/seed{seed}/bs{bs}_lr{lr}"
                        log = os.path.join(output_dir, f"{model}-slurm.log")
                        if task != "hellaswag":
                            acc = extract(log)
                        else:
                            acc = extract_hellaswag(os.path.join(output_dir, model, f"p{pattern}-i0"))
                        all_ex_count += 1
                        if acc is not None:
                            all_acc.append(acc)
                            valid_ex_count += 1
                        else:
                            run_ex(output_dir, task, bs, lr, pattern, seed, model)
                        
            try:
                best_acc.append(max(all_acc))
            except:
                continue
        if len(best_acc) == 0:
            continue
        print(f"{np.mean(best_acc)*100:.1f} {np.std(best_acc)*100:.1f}")
    print(f"{all_ex_count - valid_ex_count}/{all_ex_count}")

