# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.
import pdb
import sys

import datasets
import fire
import json
import numpy as np
import contextlib
from collections import defaultdict
import random 
from datasets import Dataset


downstream_tasks = ["sst2", "sst5", "imdb", "mr", "mnli", "qnli", "rte", "snli", "boolq", "ag_news", "copa", "storycloze", "hellaswag", "piqa"]
# from fairseq.data_utils
@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def get_random_subset(dataset, train_size, seed=42, label_name="label", exclude_idxes=None):
    def random_subset(dataset, k, candidate=None, exclude_idx=None):
        dataset_for_candidate = dataset.filter(lambda x: x[label_name] == candidate)
        all_idxes = set(range(len(dataset_for_candidate)))
        if exclude_idx is not None:
            all_idxes = all_idxes - set(exclude_idx)
        idx = np.random.choice(list(all_idxes), k, replace=False)
        print(f"Chosen idx: {idx.tolist()}")
        return idx, dataset_for_candidate.select(idx)

    candidates = dataset.unique("label")
    candidates.sort()
    with numpy_seed(seed):
        few_shot_datasets = []
        idxes = []
        for i, candidate in enumerate(candidates):
            exclude_idx = None
            if exclude_idxes is not None:
                exclude_idx = exclude_idxes[i]
            candidate_idx, candidate_few_shot_dataset = random_subset(dataset, k=train_size, candidate=candidate, exclude_idx=exclude_idx)
            few_shot_datasets.append(candidate_few_shot_dataset)
            train_dataset = datasets.concatenate_datasets(few_shot_datasets)
            idxes.append(candidate_idx)
        train_dataset = train_dataset.shuffle()
        return idxes, train_dataset


def convert_label(examples):
    lens = len(examples["label"])
    new_labels = [int(examples["label"][i]) for i in range(lens)]
    examples["label"] = new_labels
    return examples

def fix_datasets(d_name, raw_datasets, additional_args):
    if d_name == "trec":
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].add_column("label", raw_datasets[key]["label-coarse"])
        raw_datasets["validation"] = raw_datasets["test"]
    if d_name == "ag_news":
        raw_datasets["validation"] = raw_datasets["test"]
        raw_datasets = raw_datasets.map(convert_label, batched=True)
    if d_name == "sst5":
        raw_datasets = raw_datasets.map(convert_label, batched=True)
    if d_name == "hellaswag":
        if "test" in raw_datasets:
            raw_datasets.pop("test")
        raw_datasets = raw_datasets.map(convert_label, batched=True)
    if d_name == "imdb":
        raw_datasets["validation"] = raw_datasets["test"]
    if d_name == "yelp_polarity":
        raw_datasets["validation"] = raw_datasets["test"]
    if d_name == "snli":
        raw_datasets = raw_datasets.filter(lambda x: x["label"] != -1)
        raw_datasets["train"] = raw_datasets["train"].flatten_indices()
    if d_name == "boolq":
        def boolq_outer(examples):
            labels = []
            for i in range(len(examples["answer"])):
                if examples["answer"][i]:
                    label = 1
                else:
                    label = 0
                labels.append(label)
            examples.update({"label": labels})
            return examples
        raw_datasets = raw_datasets.map(boolq_outer, batched=True, remove_columns=["answer"])
    if d_name.startswith("copa-v"):
        if additional_args.from_nonsym_to_sym_dis:
            for split in raw_datasets:
                lens = len(raw_datasets[split]) // 2
                base = np.arange(lens) * 2
                sample = np.random.randint(0, 2, lens)
                raw_datasets[split] = raw_datasets[split].select(base+sample)
                assert len(raw_datasets[split]) == lens
    if d_name == "storycloze":
        for split in raw_datasets:
            raw_datasets[split] = raw_datasets[split].add_column("label", [int(a) - 1 for a in raw_datasets[split]["AnswerRightEnding"]])
    if d_name in downstream_tasks and "fewshot_validation" not in raw_datasets:
        if d_name == "mnli":
            raw_datasets["fewshot_validation_matched"] = raw_datasets.pop("validation_matched")
            raw_datasets["fewshot_validation_mismatched"] = raw_datasets.pop("validation_mismatched")
        else:
            raw_datasets["fewshot_validation"] = raw_datasets.pop("validation")
    return raw_datasets


# train dataset contains one example
def get_static_input(tokenizer, template, train_dataset, max_seq_length, objective, num_labels):
    if objective != "mlm":
        return {}
    mask_idx = tokenizer.encode(tokenizer.mask_token)[1]
    prompt_dataset = template.encode_with_label_words(train_dataset.select([0]))
    sentence1_key = "prompt" if "prompt" in prompt_dataset.features else None
    tokenized_dataset = prompt_dataset.map(lambda x: tokenize_and_mapping(x, template, tokenizer, max_seq_length, objective, sentence1_key=sentence1_key, num_labels=num_labels), batched=True, load_from_cache_file=False)
    inspect_indexes = []
    for example in tokenized_dataset:
        tokenized_prompt = example["input_ids"]
        token_start = example["token_start"][0]
        inspect_indexes.append(tokenized_prompt[token_start])
    static_input = {"mask_idx": mask_idx, "inspect_indexes": inspect_indexes}
    print(f"Inspect indexes: {inspect_indexes}")
    return static_input

def read_tsv_file(filepath, encoding="utf-8-sig"):
    data = []
    with open(filepath, "r", encoding=encoding) as f:
        keys = f.readline().strip("\n") # storycloze bug
        keys = keys.split("\t")
        if len(keys) == 1:
            keys = keys[0].split("  ")
        for i, line in enumerate(f):
            values = line.strip("\n").split("\t")
            if len(keys) != len(values):
                continue
            data.append({key: value for key, value in zip(keys, values)})
    return data

def load_from_jsonl(filepath):
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# [{a:, b:}, {a:, b:]] => {a: [], b: []}
def convert_tsv_data_to_dict(data):
    dictt = defaultdict(list)
    for d in data:
        for key in d:
            dictt[key].append(d[key])
    return dictt

# [(0, 0), (0, 4), (4, 8), (9, 11), (12, 17), (18, 25), (25, 26), (27, 28), (29, 33), (34, 38), (39, 49), (50, 55), (56, 59), (60, 67), (67, 68), (0, 0)]
# [0, 26412, 35073, 7, 97, 6173, 6, 38, 33, 4276, 4496, 59, 5, 544, 4, 2]
# Contrary to other reviews, I have zero complaints about the service.
def turn_char_to_token_boundary(offset, char_starts, char_ends):
    def inner(o, s, e):
        num_tokens = len(o) + 2
        o = np.array(o)
        o = o[o.sum(-1) > 0]
        offset_starts = o[:, 0]
        offset_ends = o[:, 1]

        span_starts = []
        span_ends = []
        for i, (char_start, char_end) in enumerate(zip(s, e)):
            start_indexes = np.where(offset_starts <= char_start)[0]
            end_indexes = np.where(offset_ends >= char_end)[0]

            if len(start_indexes) == 0 or len(end_indexes) == 0:
                span_starts.append(None)
                span_ends.append(None)
                continue

            end = end_indexes[0]  # right on the span end
            start = start_indexes[-1]

            start += 1
            end += 1

            span_starts.append(start)
            span_ends.append(end)
            assert start > 0, start
            assert end < num_tokens - 1, end
        return span_starts, span_ends

    if isinstance(char_starts[0], list):
        span_starts = []
        span_ends = []
        for o, s, e in zip(offset, char_starts, char_ends):
            span_start, span_end = inner(o, s, e)
            span_starts.append(span_start)
            span_ends.append(span_end)
    else:
        span_starts, span_ends = inner(offset, char_starts, char_ends)
    return span_starts, span_ends

def tokenize_and_mapping(examples, template, tokenizer, max_seq_length, objective="mlm", sentence1_key=None, sentence2_key=None, padding="max_length", num_labels=2):
    """
    :param examples: output from `encode` or `encode_with_label_words`
    :param template: task specific template
    :param tokenizer: huggingface tokenizer
    :param max_seq_length: max sequence length of the model
    :param sentence1_key: input keys in examples
    :param sentence2_key: input keys in examples
    :return:
        input_ids and attention_mask
        if "char_start" and "char_end" are in examples, include "token_start" and "token_end" into the returned results
    """
    if "complete" in examples:
        truncatable = examples["truncatable"]
        complete = examples["complete"]
        char_starts, char_ends = None, None
        if "char_start" in examples:
            char_starts = examples["char_start"]
            char_ends= examples["char_end"]
        tokenized_truncatable = tokenizer(truncatable, padding=False, max_length=max_seq_length, truncation=True, return_offsets_mapping=True)
        tokenized_complete = tokenizer(complete, padding=False, max_length=max_seq_length, truncation=True, return_offsets_mapping=True)
        truncatable_input_ids = tokenized_truncatable["input_ids"]
        complete_input_ids = tokenized_complete["input_ids"]
        complete_offset_mapping = tokenized_complete["offset_mapping"]
        assert template is not None
        # when combining, turn char start end to token level
        token_starts, token_ends = None, None
        if char_starts is not None:
            token_starts, token_ends = turn_char_to_token_boundary(complete_offset_mapping, char_starts, char_ends)
        input_ids, attention_masks = template.combine(truncatable_input_ids,
                                                      complete_input_ids,
                                                      max_seq_length,
                                                      padding=tokenizer.encode(tokenizer.pad_token)[1],
                                                      token_starts=token_starts,
                                                      token_ends=token_ends)
        result = {"input_ids": input_ids, "attention_mask": attention_masks}
        if token_starts is not None:
            result.update({"token_start": token_starts, "token_end": token_ends})
    else:
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
        )
        return_offsets_mapping = "char_start" in examples
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True, return_offsets_mapping=return_offsets_mapping)
        if return_offsets_mapping:
            offset_mapping = result["offset_mapping"]
            char_starts = examples["char_start"]
            char_ends = examples["char_end"]

            token_starts, token_ends = turn_char_to_token_boundary(offset_mapping, char_starts, char_ends)

            # sometimes text is too long and got truncated, ignore the examples where the span is trucntrued
            filter_index = [token_starts[i][0] is not None and token_ends[i][0] is not None for i in range(len(token_starts))]
            # sometimes does not work with num_proc=20
            if objective == "dis":
                for i in range(0, len(filter_index), num_labels):
                    if False in filter_index[i: i+num_labels]:
                        for k in range(i, i+num_labels):
                            filter_index[k] = False
            
            for key in result:
                result[key] = [result[key][i] for i in range(len(token_starts)) if filter_index[i]]
            for key in examples.data.keys():
                result[key] = [examples[key][i] for i in range(len(token_starts)) if filter_index[i]]

            result["token_start"] = [token_starts[i] for i in range(len(filter_index)) if filter_index[i]]
            result["token_end"] = [token_ends[i] for i in range(len(filter_index)) if filter_index[i]]
        examples.update(result)
        result = examples
    return result

def grep_best_performance(file):
    lines = open(file, "r").readlines()
    lines = [eval(line) for line in lines if line.startswith("{'") and "eval_loss" in line]
    best_accuracy = 0
    for line in lines:
        if line["eval_accuracy"] > best_accuracy:
            best_accuracy = line["eval_accuracy"]
    print(round(best_accuracy * 100, 2))

from datasets import load_from_disk
def print_prompt(file):
    d = load_from_disk(file)
    for i in range(100):
        print(d[i]["prompt"])

def fix_config(config, additional_args):
    # when loading a trained model using its own span_rep_type
    if not hasattr(config, "span_rep_type"):
        config.span_rep_type = additional_args.span_rep_type
    if not hasattr(config, "discriminator_head"):
        config.discriminator_head = additional_args.discriminator_head

def sanity_check_for_texts(dataset1, dataset2, key1, key2):
    d1_text = dataset1[key1]
    d2_text = dataset2[key1]
    lens = len(d1_text) + len(d2_text)
    if key2 is not None:
        d1k2 = dataset1[key2]
        d2k2 = dataset2[key2]
        d1_text = [a + b for a, b in zip(d1_text, d1k2)]
        d2_text = [a + b for a, b in zip(d2_text, d2k2)]
    merge_lens = len(set(d1_text).union(set(d2_text)))
    assert lens == merge_lens
    
def load_record_dataset(file, max_train_candidates_per_question=10):
    entity_shuffler = random.Random(10)
    data_dict = {"text": [], "question": [], "passage_idx": [], "question_idx": [], "candidates": [], "answers": []}
    with open(file, encoding='utf8') as f:
        for idx, line in enumerate(f):
            example_json = json.loads(line)

            idx = example_json['idx']
            text = example_json['passage']['text']
            entities = set()

            for entity_json in example_json['passage']['entities']:
                start = entity_json['start']
                end = entity_json['end']
                entity = text[start:end + 1]
                entities.add(entity)

            entities = list(entities)

            text = text.replace("@highlight\n", "- ")  # we follow the GPT-3 paper wrt @highlight annotations
            questions = example_json['qas']

            for question_json in questions:
                question = question_json['query']
                question_idx = question_json['idx']
                answers = set()

                for answer_json in question_json.get('answers', []):
                    answer = answer_json['text']
                    answers.add(answer)

                answers = list(answers)

                
                for answer_idx, answer in enumerate(answers):
                    candidates = [ent for ent in entities if ent not in answers]
                    if len(candidates) > max_train_candidates_per_question - 1:
                        entity_shuffler.shuffle(candidates)
                        candidates = candidates[:max_train_candidates_per_question - 1]
                    data_dict["text"].append(text)
                    data_dict["question"].append(question)
                    data_dict["passage_idx"].append(idx)
                    data_dict["question_idx"].append(question_idx)
                    data_dict["candidates"].append([answer] + candidates)
                    data_dict["answers"].append(answer)
    dataset = Dataset.from_dict(data_dict)
    return dataset

if __name__ == '__main__':

    fire.Fire()
    sys.exit()
    # test
    offset = [(0, 0), (0, 4), (4, 8), (9, 11), (12, 17), (18, 25), (25, 26), (27, 28), (29, 33), (34, 38), (39, 49), (50, 55), (56, 59), (60, 67), (67, 68), (0, 0)]
    tokens = [0, 26412, 35073, 7, 97, 6173, 6, 38, 33, 4276, 4496, 59, 5, 544, 4, 2]
    text = "Contrary to other reviews, I have zero complaints about the service."

    char_starts = [12, 39]
    char_ends = [17, 49]
    span_starts, span_ends = turn_char_to_token_boundary(offset, char_starts, char_ends)
    assert span_starts == [4, 10]
    assert span_ends == [4, 10]
