
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.import os
import sys
import pdb
from click import prompt
import torch
import numpy as np
import nltk
from datasets import load_dataset, load_from_disk, DatasetDict
from keyword_process import basics_dataset
import random
from collections import defaultdict

def load_retrieval_results():
    train_file = os.path.join(dir, "sents.train.valid.reps.retrieval", "train.retrieval.k50")
    valid_file = os.path.join(dir, "sents.train.valid.reps.retrieval", "valid.retrieval.k50")
    train_retrieved_results = torch.load(train_file)
    valid_retrieved_results = torch.load(valid_file)
    return {"train": train_retrieved_results, "valid": valid_retrieved_results}

def get_subline(sent):
    if " so " in sent or " because " in sent:
        if " so " in sent:
            index = sent.index(" so ")
        elif " because " in sent:
            index = sent.index(" because ")
        subline = sent[index:].strip()
        try:
            subline = subline # truecase is problematic, it adds space to some cases
            if subline != "":
                return subline
        except:
            return None
    return None

def process_lines_with_texts(examples):
        lines = examples["text"]
        re = {"subline": [], "char_start": [], "char_end": [], "text": []}
        for i, line in enumerate(lines):
            line = line.strip()
            sents = nltk.sent_tokenize(line)
            for i, sent in enumerate(sents):
                subline = get_subline(sent)
                if subline is not None:
                    re["subline"].append(subline)
                    char_start = line.index(subline)
                    char_end = char_start + len(subline)
                    re["char_start"].append([char_start])
                    re["char_end"].append([char_end])
                    re["text"].append(line)
                    assert subline == line[char_start: char_end]
                    break
            if len(re["subline"]) == i:
                re["subline"].append(None)
                re["char_start"].append([])
                re["char_end"].append([])
                re["text"].append(None)
        return re

# retrieval_texts: list of texts
# scores: np array of scores for k returned texts
def select_retrieval(retrieval_texts, scores, indexes, threshold=None):
    if threshold is None:
        i = np.where(scores > 0)[0][0]
        score = scores[i]
        index = indexes[i]
        return retrieval_texts[index]
    else:
        i = np.where(scores > threshold)[0]
        if len(i) == 0:
            i = -1
        else:
            i = i[0]
        score = scores[i]
        index = indexes[i]
        return retrieval_texts[index], score

def get_replaced_texts(examples, indices, retrieval_results, query_texts, threshold):
    replaced_texts = []
    scores = []
    for indice in zip(indices):
        retrieval_index, retrieval_score = retrieval_results["index"][indice], retrieval_results["score"][indice]
        t, s = select_retrieval(query_texts, retrieval_score, retrieval_index, threshold)
        replaced_texts.append(t.strip())
        scores.append(round(s, 5))
    return {"replaced_texts": replaced_texts, "replaced_texts_scores": scores}

def print_dataset_processing_info(dataset, info):
    if not isinstance(dataset, DatasetDict):
        print(info.format("dataset", len(dataset)))
    else:
        for cate in dataset.keys():
            print(info.format(cate, len(dataset[cate]))) 

def process_random_lines():
    file = os.path.join(dir, "sents.random")
    raw_dataset = load_dataset("text", data_files=file, encoding='ISO-8859-1')
    print_dataset_processing_info(raw_dataset, "{}: Loaded {} examples")
    
    processed_dataset = raw_dataset.map(process_lines_with_texts, batched=True, num_proc=20)
    print_dataset_processing_info(raw_dataset, "{}: After processing the lines with texts, there are {} examples")
    
    processed_dataset = processed_dataset.filter(lambda x: x["subline"] is not None)
    print_dataset_processing_info(processed_dataset, "{}: After filtering, there are {} examples")

    save_path = os.path.join(v2dir, "preprocess-random", "processed_dataset.pt")
    processed_dataset.save_to_disk(save_path)
    print(f"Saving the processed random dataset to {save_path}.")

    basics_dataset(processed_dataset["train"])
    
# preprocess_raw_datasets
def step1():
    train_raw_file = os.path.join(dir, "sents.all.shuf.train")
    valid_raw_file = os.path.join(dir, "sents.all.shuf.valid")
    print(train_raw_file)
    data_file = {"train": train_raw_file, "valid": valid_raw_file}
    raw_dataset = load_dataset("text", data_files=data_file, encoding='ISO-8859-1')
    print_dataset_processing_info(raw_dataset, "{}: Loaded {} examples")

    
    processed_dataset = raw_dataset.map(process_lines_with_texts, batched=True, num_proc=20)
    print_dataset_processing_info(raw_dataset, "{}: After processing the lines with texts, there are {} examples")
    
    processed_dataset = processed_dataset.filter(lambda x: x["subline"] is not None)
    print_dataset_processing_info(processed_dataset, "{}: After filtering, there are {} examples")

    save_path = os.path.join(v2dir, "preprocess", "processed_dataset.pt")
    processed_dataset.save_to_disk(save_path)
    print(f"Saving the processed query dataset to {save_path}.")

    basics_dataset(processed_dataset["train"])
    basics_dataset(processed_dataset["valid"])

def step2(n_tokens, version=2):
    def split_sent(examples):
        lens = len(examples["subline"])
        heads = []
        for i in range(lens):
            head = " ".join(examples["subline"][i].split()[:n_tokens])
            heads.append(head)
        examples["head"] = heads 
        return examples
    
    dataset = load_from_disk(os.path.join(v2dir, "preprocess", "processed_dataset.pt"))
    print_dataset_processing_info(dataset, "{}: Loaded {} examples")
    dataset = dataset.map(split_sent, batched=True, num_proc=20)
    
    def cate(dataset):
        d =  defaultdict(list)
        sublimes = dataset["subline"]
        heads = dataset["head"]
        for sublime, head in zip(sublimes, heads):
            d[head].append(sublime)
        return d
    
    if version == 2:
        classified_subline = cate(dataset["train"])
        valid_classified_subline = cate(dataset["valid"])
        for key in valid_classified_subline:
            classified_subline[key].extend(valid_classified_subline[key])
    elif version == 3:
        random_subline_dataset = load_from_disk(os.path.join(v2dir, "preprocess-random", "processed_dataset.pt"))
        random_subline_dataset = random_subline_dataset.map(split_sent, batched=True, num_proc=20)
        classified_subline = cate(random_subline_dataset["train"])
        
    

    for key in classified_subline:
        classified_subline[key] = list(set(classified_subline[key]))
    
    other_keys = []
    for key in classified_subline:
        if len(classified_subline[key]) < 10:
            other_keys.append(key)
    
    other = []
    for key in other_keys:
        other.extend(classified_subline.pop(key))

    if len(other) > 0:
        classified_subline["other"] = other
    
    subline_statistics = {key: len(classified_subline[key]) for key in classified_subline}
    print("Classified Subline Statistics:", subline_statistics)

    def find_replacement(examples):
        lens = len(examples["subline"])
        replaced_sublines = []
        for i in range(lens):
            subline = examples["subline"][i]
            head = examples["head"][i]
            replaced_subline = subline
            while subline == replaced_subline:
                if head in classified_subline:
                    replaced_subline = random.choice(classified_subline[head])
                else:
                    random_head = random.choice(list(classified_subline.keys()))
                    replaced_subline = random.choice(classified_subline[random_head])
            assert subline != replaced_subline
            replaced_sublines.append(replaced_subline)
        examples["replaced_subline"] = replaced_sublines
        return examples

    dataset = dataset.map(find_replacement, batched=True, num_proc=20)
    outdir = os.path.join(v2dir, "replacement", f"ntokens{n_tokens}")
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "dataset.pt")
    
    dataset.save_to_disk(outfile)
    print_dataset_processing_info(dataset, "{}: Saving to {} examples to " + outfile)
    basics_dataset(dataset["train"])
    basics_dataset(dataset["valid"])

def step3(n_tokens):
    file = os.path.join(v2dir, "replacement", f"ntokens{n_tokens}", "dataset.pt")
    dataset = load_from_disk(file)
    print_dataset_processing_info(dataset, "{}: Loaded {} examples")
    
    def replace_subline_in_text(text, char_start, char_end, replaced_subline):
        new_text = text[:char_start]
        new_text += replaced_subline
        new_char_end = len(new_text)
        new_text += text[char_end:]
        return new_text, char_start, new_char_end
    
    def exclude_so_because(text, char_start, char_end):
        subline = text[char_start: char_end]
        if subline.startswith("so"):
            char_start += 3
        elif subline.startswith("because"):
            char_start += 8
        return char_start

    def transform_to_prompt_dataset(examples):
        prompts = []
        char_starts = []
        char_ends = []
        labels = []
        lens = len(examples["subline"])
        for i in range(lens):
            text = examples["text"][i]
            replaced_subline = examples["replaced_subline"][i]
            char_start = examples["char_start"][i][0]
            char_end = examples["char_end"][i][0]
            tweak_char_start = exclude_so_because(text, char_start, char_end)

            prompts.append(text)
            char_starts.append([tweak_char_start])
            char_ends.append([char_end])
            labels.append(0)

            new_text, new_char_start, new_char_end = replace_subline_in_text(text, char_start, char_end, replaced_subline)
            tweak_new_char_start = exclude_so_because(text, new_char_start, new_char_end)
            prompts.append(new_text)
            char_starts.append([tweak_new_char_start])
            char_ends.append([new_char_end])
            labels.append(1)
        return {"prompt": prompts, "char_start": char_starts, "char_end": char_ends, "label": labels}
    
    dataset = dataset.map(transform_to_prompt_dataset, batched=True, remove_columns=['head', 'replaced_subline', 'subline', 'text'])
    
    output_dir = os.path.join(v2dir, "prompt_dataset", f"ntokens{n_tokens}")
    dataset["train"].flatten_indices().save_to_disk(os.path.join(output_dir, "train.pt"))
    dataset["valid"].flatten_indices().save_to_disk(os.path.join(output_dir, "valid.pt"))

    basics_dataset(dataset["train"])
    basics_dataset(dataset["valid"])


if __name__ == '__main__':
    dir = "/n/fs/nlp-mengzhou/space3/meta/data/copa.bin"
    v2dir ="/n/fs/nlp-mengzhou/space3/data/created/copa-v3"
    func = sys.argv[1] 

    if func == "preprocess_raw_datasets":
        step1()
    if func == "find_replacement":
        n_tokens = int(sys.argv[2]) 
        step2(n_tokens=n_tokens, version=3)
    if func == "get_prompt":
        n_tokens = int(sys.argv[2]) 
        step3(n_tokens=n_tokens)
    if func == "process_random_lines":
        process_random_lines()

    