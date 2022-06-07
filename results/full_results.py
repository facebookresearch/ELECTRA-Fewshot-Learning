# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np


def extract(file):
    try:
        lines = open(file, "r").readlines()
        for line in lines[::-1]:
            if "fewshot_validation" in line:
                acc = float(line.split(" ")[-1])
                return acc
    except:
        return None
    return None

re = {}
    
for task in ["sst2", "snli", "ag_news", "boolq", "sst5", "mr", "qnli", "mnli", "rte"]:
    for model in ["bert-base-uncased", "roberta-base", "google/electra-base-discriminator"]:
        if model not in re:
            re[model] = []
        accs = []
        for bs in [16, 32]:
            for lr in ["1e-5", "2e-5", "3e-5"]:
                output_dir = f"/n/fs/nlp-mengzhou/space3/out/hf/FT/{task}/{model}/bs{bs}_lr{lr}"
                log_file = os.path.join(output_dir, "log.out")
                acc = extract(log_file)
                if acc is not None:
                    accs.append(acc)
        accs.pop(np.argmin(accs))
        re[model].append(np.mean(accs))
        print(f"\\textit{{ {round(np.mean(accs) * 100, 1)} ({round(np.std(accs) * 100, 1)}) }} & ", end="")

bert = np.mean(re["bert-base-uncased"])
roberta = np.mean(re["roberta-base"])
electra = np.mean(re["google/electra-base-discriminator"])
print(electra - bert)
print(electra-roberta)