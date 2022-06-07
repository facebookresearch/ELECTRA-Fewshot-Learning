# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
    python3 ~/fairseq-py/examples/few_shot/few_shot_discriminator/pretrain/heuristic/keyword.py
"""
import pdb
import sys
import string
import datasets
from datasets import load_dataset
import numpy as np
import os
import torch
from copy import deepcopy

remove_comma_tasks = ["nli", "boolq"]

def load_raw_data(file, cache_dir):
    data_files = {"eval": file}
    try:
        dataset = load_dataset("text", data_files=data_files, cache_dir=cache_dir, encoding='utf-8')
    except:
        dataset = load_dataset("text", data_files=data_files, cache_dir=cache_dir, encoding="ISO-8859-1")
    return dataset["eval"]



def filter_keyword(text, keyword_in_context, prev_text=None, include_prev_sent=False):
    filtered_text = None
    text = text.strip()
    if text != "":
        if keyword_in_context in text:
            yes = True
            if include_prev_sent:
                if prev_text is not None:
                    filtered_text = f"{prev_text.strip()} {text}"
            else:
                filtered_text = text
    return filtered_text


def filter_keyword_examples(examples, keyword_in_context, include_prev_sent=False):
    """
    :param examples: each example contains a column `text`
    :param keyword_in_context: a list of keywords
    :return:
        {"text": [sentences that contain at least one keyword]}
    """
    lens = len(examples["text"])
    filtered_texts = []
    for i in range(lens):
        text = examples["text"][i].strip()
        prev_text = None
        if i > 0:
            prev_text = examples["text"][i-1].strip()
        filtered_text = filter_keyword(text, keyword_in_context, prev_text, include_prev_sent)
        if filtered_text is not None:
            filtered_texts.append(filtered_text)
    return {"text": filtered_texts}


def find_keyword(text, keyword_in_context, keyword_in_tokens):
    # find the keyword and return the token level span starts and ends
    char_starts = []
    char_ends = []

    st = keyword_in_context.index(keyword_in_tokens)
    en = st + len(keyword_in_tokens)
    keyword_head = keyword_in_context[:st]
    keyword_tail = keyword_in_context[en:]
    keyword_in_context_length = len(keyword_in_context)

    assert keyword_in_context in text
    index = text.index(keyword_in_context)

    sent = text[:index]
    sent += keyword_head

    char_start = len(sent)
    sent += keyword_in_tokens
    char_end = len(sent)

    sent += keyword_tail
    sent += text[index + keyword_in_context_length:]

    char_starts.append(char_start)
    char_ends.append(char_end)
    keyword = [sent[char_start: char_end]]
    assert keyword[0] == keyword_in_tokens
    return sent, char_starts, char_ends, keyword

def find_keyword_examples(examples, keyword_in_context, keyword_in_tokens):
    lens = len(examples["text"])
    new_texts, char_starts, char_ends, keywords = [], [], [], []
    for i in range(lens):
        text = examples["text"][i].strip()
        sent, char_start, char_end, keyword = find_keyword(text, keyword_in_context, keyword_in_tokens)
        char_starts.append(char_start)
        char_ends.append(char_end)
        new_texts.append(sent)
        keywords.append(keyword)
    return {"text": new_texts, "char_start": char_starts, "char_end": char_ends, "keyword": keywords}

def sort_multiple_by_one(*lists):
    # sort all lists based on the order of the first list
    main_list = lists[0]
    index = np.argsort(main_list)
    sorted_lists = []
    for list in lists:
        sorted_lists.append(np.array(list)[index])
    return sorted_lists


def from_keyword_to_final_word(final_words_to_keywords, keyword):
    for final_word in final_words_to_keywords:
        if keyword in final_words_to_keywords[final_word]:
            return final_word
    print("Something went wrong")
    pdb.set_trace()


def unify_final_words(text, keywords, char_starts, char_ends, final_words_to_keywords, final_words, obj, remove_comma=False):
    new_char_starts, new_char_ends, new_sents, new_keywords, new_labels = [], [], [], [], []
    for j, (st, en) in enumerate(zip(char_starts, char_ends)):
        prev_text = text[: st]
        next_text = text[en:]
        cur_keyword = text[st: en]
        assert cur_keyword == keywords[j]

        cur_final_word = from_keyword_to_final_word(final_words_to_keywords, cur_keyword)

        if obj == "dis":
            for final_word in final_words_to_keywords:
                char_start = len(prev_text)
                new_sent = prev_text + final_word
                char_end = len(new_sent)
                new_sent += next_text
                if final_word == cur_final_word:
                    new_label = 0
                else:
                    new_label = 1
                new_char_starts.append([char_start])
                if remove_comma:
                    char_end -= 1
                new_char_ends.append([char_end])
                new_sents.append(new_sent)
                new_keywords.append([final_word])
                new_labels.append(new_label)
        elif obj == "mlm":
            char_start = len(prev_text)
            new_sent = prev_text + "<mask>"
            char_end = len(new_sent)

            if remove_comma:
                new_sent += ","
            new_sent += next_text
            new_label = final_words.index(cur_final_word)
            final_word = "<mask>"
            new_char_starts.append([char_start])
            new_char_ends.append([char_end])
            new_sents.append(new_sent)
            new_keywords.append([final_word])
            new_labels.append(new_label)
        else:
            print(f"{obj} mode is not supported.")
            sys.exit()


    return new_char_starts, new_char_ends, new_sents, new_keywords, new_labels


def unify_final_words_examples(examples, final_words_to_keywords, final_words, obj="dis", remove_comma=False):
    lens = len(examples["text"])
    new_char_starts = []
    new_char_ends = []
    new_keywords = []
    new_labels = []
    new_texts = []

    for i in range(lens):
        char_starts = examples["char_start"][i]
        char_ends = examples["char_end"][i]
        keywords = examples["keyword"][i]
        text = examples["text"][i]
        cs, ce, ns, nk, nl = unify_final_words(text, keywords, char_starts, char_ends, final_words_to_keywords, final_words, obj, remove_comma)
        new_char_starts.extend(cs)
        new_char_ends.extend(ce)
        new_texts.extend(ns)
        new_keywords.extend(nk)
        new_labels.extend(nl)

    words = []
    for st, en, text in zip(new_char_starts, new_char_ends, new_texts):
        words.append(text[st[0]: en[0]])
    if object == "mlm":
        assert len(set(words)) == 1
    elif object == "dis":
        assert len(set(words)) == len(final_words)

    examples.update({"char_start": new_char_starts, "char_end": new_char_ends, "keyword": new_keywords,
                     "label": new_labels, "prompt": new_texts})
    return examples


def basics_dataset(dataset):
    features = list(dataset.features.keys())
    print("****** Dataset Inspection ******")
    print(f"There are {len(dataset)} examples in the dataset.")
    print(f"Features: {features}")
    print(f"Example: {dataset[0]}")



def get_words_mapping():
    # directory_keword, keyword in text, keywords, whether to include the previous sentence
    a = {"good": [[" good "] + ["good" + s for s in ".?!,;:'"], "good", False]}
    a.update({"okay": [[" okay "] + ["okay" + s for s in ".?!,;:'"], "okay", False]})
    a.update({"bad": [[" bad "] + ["bad" + s for s in ".?!,;:'"], "bad", False]})
    a.update({"great": [[" great "] + ["great" + s for s in ".?!,;:'"], "great", False]})
    a.update({"great-v1": [["It was great "] + ["It was great" + s for s in ".?!,;:'"], "It was great", True]})
    a.update({"great-v2": [["It is great "] + ["It is great" + s for s in ".?!,;:'"], "It is great", True]})
    a.update({"great-v3": [["it was great "] + ["it was great" + s for s in ".?!,;:'"], "it was great", True]})
    a.update({"great-v4": [["it is great "] + ["it is great" + s for s in ".?!,;:'"], "it is great", True]})
    a.update({"terrible": [[" terrible "] + ["terrible" + s for s in ".?!,;:'"], "terrible", False]})
    a.update({"terrible-v1": [["It was terrible "] + ["It was terrible" + s for s in ".?!,;:'"], "It was terrible", True]})
    a.update({"terrible-v2": [["It is terrible "] + ["It is terrible" + s for s in ".?!,;:'"], "It is terrible", True]})
    a.update({"terrible-v3": [["it was terrible "] + ["it was terrible" + s for s in ".?!,;:'"], "it was terrible", True]})
    a.update({"terrible-v4": [["it is terrible "] + ["it is terrible" + s for s in ".?!,;:'"], "it is terrible", True]})
    a.update({"which_means": [[" which means "], "which means", False]})
    a.update({"which_implies": [[" which implies "], "which implies", False]})
    a.update({"Similarly-v1": [["Similarly "], "Similarly", True]})
    a.update({"Similarly-v2": [["Similarly,"], "Similarly,", True]})
    a.update({"In_contrast": [["In contrast, "], "In contrast,", True]})
    a.update({"On_the_contrary": [["On the contrary, "], "On the contrary,", True]})
    a.update({"Yes-v1": [["? Yes,"], "Yes,", False]})
    a.update({"Yes-v2": [[f"{i} Yes," for i in ".!;"], "Yes,", False]})
    a.update({"No-v1": [["? No,"], "No,", False]})
    a.update({"No-v2": [[f"{i} No," for i in ".!;"], "No,", False]})
    a.update({"Maybe-v1": [["? Maybe,"], "Maybe,", False]})
    a.update({"Maybe-v2": [[f"{i} Maybe " for i in ".!;"], "Maybe", False]})
    a.update({"as": [[" as "], "as", False]})
    a.update({"because": [[" because "], "because", False]})
    a.update({"World": [["World "], "World", False]})
    a.update({"world": [[" world "], "world", False]})
    a.update({"Business": [["Business "], "Business", False]})
    a.update({"business": [[" business "], "business", False]})
    a.update({"Sports": [["Sports "], "Sports", False]})
    a.update({"sports": [[" sports "], "sports", False]})
    a.update({"Tech": [["Tech "], "Tech", False]})
    a.update({"tech": [[" tech "], "tech", False]})


    sets = []
    sets.append(["good", "okay", "bad", "great", "terrible"])
    sets.append(["great", "terrible"])
    sets.append(["Yes,", "No,", "Maybe,"])
    sets.append(["as", "because"])
    sets.append(["person", "location", "number", "description", "entity", "abbreviation"])
    return a

def sst5_template(test=False):
    final_words = ["good", "okay", "bad", "great", "terrible"]
    final_words_to_keywords = {w: [w] for w in final_words}
    directory_keywords = final_words
    nums = [90000] * 5
    return final_words, final_words_to_keywords, directory_keywords, nums

def agnews_capital_template(test=False):
    final_words = ["World", "Sports", "Business", "Tech"]
    final_words_to_keywords = {w: [w] for w in final_words}
    directory_keywords = final_words
    nums = [50000] * 4
    return final_words, final_words_to_keywords, directory_keywords, nums

def agnews_all_template(test=False):
    final_words = ["world", "sports", "business", "tech"]
    final_words_to_keywords = {w: [w, w.capitalize()] for w in final_words}
    directory_keywords = deepcopy(final_words)
    directory_keywords += [w.capitalize() for w in final_words]
    nums = [50000] * 8
    return final_words, final_words_to_keywords, directory_keywords, nums


def sst2_template(test=False):
    final_words = ["terrible", "great"]
    final_words_to_keywords = {w: [w] for w in final_words}
    directory_keywords = final_words
    nums = [95336, 95336]
    return final_words, final_words_to_keywords, directory_keywords, nums

def sst2_template_v2(test=False):
    final_words = ["terrible", "great"]
    final_words_to_directory_keywords = {"great": [f"great-v{i}" for i in range(1, 5)], 
                               "terrible": [f"terrible-v{i}" for i in range(1, 5)]}
    final_words_to_keywords = {key: [word_mapping[directory_keyword][1] for directory_keyword in final_words_to_directory_keywords[key]] for key in final_words_to_directory_keywords}
    directory_keywords = sum([final_words_to_directory_keywords[final_word] for final_word in final_words], [])
    nums = [660, 196, 593, 175, 660, 196, 593, 175]
    return final_words, final_words_to_keywords, directory_keywords, nums

def boolq_template(test=False):
    final_words = ["Yes,", "No,"]
    final_words_to_directory_keywords = {
        "Yes,": ["Yes-v1", "Yes-v2",],
        "No,": ["No-v1", "No-v2"]}
    final_words_to_keywords = {key: [word_mapping[directory_keyword][1] for directory_keyword in final_words_to_directory_keywords[key]] for key in final_words_to_directory_keywords}
    directory_keywords = sum([final_words_to_directory_keywords[final_word] for final_word in final_words], [])
    nums = [6501, 15000, 7183, 15000]
    return final_words, final_words_to_keywords, directory_keywords, nums


def nli_template(test=False):
    # sample from
    final_words = ["Yes,", "No,", "Maybe,"]

    # for checking label
    final_words_to_directory_keywords = {"Yes,": ["Yes-v1", "Yes-v2", "which_means", "which_implies", "Similarly-v1", "Similarly-v2"],
     "No,": ["No-v1", "No-v2", "In_contrast", "On_the_contrary",],
     "Maybe,": ["Maybe-v1", "Maybe-v2"]}

    final_words_to_keywords = {key: [word_mapping[directory_keyword][1] for directory_keyword in final_words_to_directory_keywords[key]] for key in final_words_to_directory_keywords}

    # keywords
    directory_keywords = sum([final_words_to_directory_keywords[final_word] for final_word in final_words], [])
    if test:
        nums = [3] * len(directory_keywords)
    else:
        nums = [6501, 11000, 11000, 1423, 3779, 11000, 7183, 15800, 15974, 5439, 808, 44000]
    return final_words, final_words_to_keywords, directory_keywords, nums

def nli_template_v2(test=False):
    # sample from
    final_words = ["Yes,", "No,", "Maybe,"]

    # for checking label
    final_words_to_directory_keywords = {"Yes,": ["Yes-v1", "Yes-v2"],
     "No,": ["No-v1", "No-v2"],
     "Maybe,": ["Maybe-v1", "Maybe-v2"]}

    final_words_to_keywords = {key: [word_mapping[directory_keyword][1] for directory_keyword in final_words_to_directory_keywords[key]] for key in final_words_to_directory_keywords}

    # keywords
    directory_keywords = sum([final_words_to_directory_keywords[final_word] for final_word in final_words], [])
    if test:
        nums = [3] * len(directory_keywords)
    else:
        nums = [6501, 11000, 6501, 11000, 808, 17000]
    return final_words, final_words_to_keywords, directory_keywords, nums



def copa_template(test=False):
    final_words = ["as", "because"]
    final_words_to_keywords = {w: [w] for w in final_words}
    directory_keywords = final_words
    nums = [100000, 100000]
    return final_words, final_words_to_keywords, directory_keywords, nums

def trec_template(test=False):
    final_words = ["person", "location", "number", "description", "entity", "abbreviation"]
    final_words_to_keywords = {w: [w] for w in final_words}
    directory_keywords = final_words
    nums = [2222] * 6
    return final_words, final_words_to_keywords, directory_keywords, nums

def subj_template(test=False):
    final_words = ["subjective", "objective"]
    final_words_to_keywords = {w: [w] for w in final_words}
    directory_keywords = final_words
    nums = [9073, 9073]
    return final_words, final_words_to_keywords, directory_keywords, nums

all_templates = {"sst5": sst5_template,
                 "sst2": sst2_template,
                 "copa": copa_template,
                 "trec": trec_template,
                 "nli": nli_template,
                 "nli_v2": nli_template_v2,
                 "subj": subj_template,
                 "agnews_capital": agnews_capital_template,
                 "agnews_all": agnews_all_template,
                 "boolq": boolq_template,
                 "sst2_v2": sst2_template_v2}

def get_shards():
    return [f"0{i}" for i in range(10)] + [f"{str(i)}" for i in range(10, 90)] + [str(i) for i in range(9000, 9102)]

def subj_template(test=False):
    final_words = ["subjective", "objective"]

def get_index():
    return ["0" + str(i) for i in range(0, 10)] + [str(i) for i in range(10, 90)] + [str(i) for i in range(9000, 9102)]

def step1(file, cache_dir, keyword_dataset_dir):
    raw_dataset = load_raw_data(file, cache_dir)
    words_in_context, keyword_in_tokens, include_prev_sent = word_mapping[keyword]

    print(f"Raw datasets: {len(raw_dataset)}")
    keyword_datasets = []
    for word_in_context in words_in_context:
        filter_dataset = raw_dataset.map(lambda x: filter_keyword_examples(x, word_in_context, include_prev_sent),
                                         batched=True)
        print(f"After filtering with <{word_in_context}>: {len(filter_dataset)}")
        keyword_dataset = filter_dataset.map(lambda x: find_keyword_examples(x, word_in_context, keyword_in_tokens),
                                             batched=True)  # include c
        print(f"After finding char start and end for keyword: {len(keyword_dataset)}")
        if len(keyword_dataset) > 0:
            keyword_datasets.append(keyword_dataset)
    if len(keyword_datasets) > 0:
        keyword_datasets = datasets.concatenate_datasets(keyword_datasets)
        print(f"After aggregating all keywords in context: {len(keyword_datasets)}")

        keyword_dataset_dir = os.path.join(keyword_dataset_dir, keyword)
        os.makedirs(keyword_dataset_dir, exist_ok=True)
        keyword_dataset_file = os.path.join(keyword_dataset_dir, f"train.1%{index}")
        torch.save(keyword_datasets, keyword_dataset_file)
        print(f"Saving keyword_dataset to {keyword_dataset_file} for keyword {keyword}.")
        print()

        basics_dataset(keyword_datasets)
        for i in range(min(10, len(keyword_datasets))):
            sent = keyword_datasets[i]['text']
            char_start = keyword_datasets[i]["char_start"][0]
            char_end = keyword_datasets[i]["char_end"][0]
            print(f"Sentence {i}: {sent}")
            print(f"Keyword {i}: {sent[char_start: char_end]}")
            print()

def step2():
    all_keyword_datasets = []
    for ii in get_shards():
        file = os.path.join(keyword_dataset_dir, keyword, f"train.1%{ii}")
        if os.path.exists(file):
            keyword_dataset = torch.load(file)
            if len(keyword_dataset) > 0:
                all_keyword_datasets.append(keyword_dataset)
    keyword_datasets = datasets.concatenate_datasets(all_keyword_datasets)

    # save the keyword file
    keyword_datasets_file = os.path.join(keyword_dataset_dir, keyword, "all.pt")
    torch.save(keyword_datasets, keyword_datasets_file)
    print(f"Saving all keyword_datasets to {keyword_datasets_file} for keyword {keyword}.")
    basics_dataset(keyword_datasets)

def step3(test=False):
    names = templates_names.split(",")
    templates = [all_templates[template](test=test) for template in templates_names.split(",")]
    labeled_datasets = []
    for i, template in enumerate(templates):
        remove_comma = names[i] in remove_comma_tasks
        keyword_datasets_one_template = []
        final_words, final_words_to_keywords, directory_keywords, nums = template
        for i, directory_keyword in enumerate(directory_keywords):
            specific_keyword_dataset_dir = os.path.join(keyword_dataset_dir, directory_keyword)
            keyword_dataset_file = os.path.join(specific_keyword_dataset_dir, "all.pt")
            keyword_dataset = torch.load(keyword_dataset_file)
            print(f"Loaded {directory_keyword} from {keyword_dataset_file}: {len(keyword_dataset)}")
            keyword_dataset = keyword_dataset.shuffle().select(list(range(nums[i])))
            print(f"Selected {directory_keyword}: {len(keyword_dataset)}")
            keyword_datasets_one_template.append(keyword_dataset)
        keyword_datasets_one_template = datasets.concatenate_datasets(keyword_datasets_one_template)
        unified_dataset = keyword_datasets_one_template.map(
            lambda x: unify_final_words_examples(x, final_words_to_keywords, final_words, object, remove_comma), batched=True, desc="unifying label words",
                                                 remove_columns=["text"])
        labeled_datasets.append(unified_dataset)

    labeled_datasets = datasets.concatenate_datasets(labeled_datasets)
    if not test:
        valid_num = 500
        if object == "dis":
            num_labels = len(final_words)
            index = np.random.permutation(len(labeled_datasets) // num_labels)[:valid_num] * num_labels
            index = np.stack([index+i for i in range(num_labels)]).transpose().reshape(-1)
            valid_dataset = labeled_datasets.select(index)
            train_index = set(range(len(labeled_datasets))) - set(index)
            train_dataset = labeled_datasets.select(train_index)
            lens = len(valid_dataset["label"])
            assert lens / (lens - sum(valid_dataset["label"]))  == num_labels
        else:
            labeled_datasets = labeled_datasets.train_test_split(test_size=valid_num)
            train_dataset = labeled_datasets["train"]
            valid_dataset = labeled_datasets["test"]

        outfile = f"{aggregated_dataset_dir}/{templates_names}_aug/train.pt"
        train_dataset.flatten_indices().save_to_disk(outfile)
        print(f"Saving training dataset to {outfile}: {len(train_dataset)} examples.")

        outfile = f"{aggregated_dataset_dir}/{templates_names}_aug/valid.pt"
        valid_dataset.flatten_indices().save_to_disk(outfile)
        print(f"Saving validation dataset to {outfile}: {len(valid_dataset)} examples.")

        basics_dataset(train_dataset)
        basics_dataset(valid_dataset)
    else:
        basics_dataset(labeled_datasets)

"""
    python3 ~/hf/data/keyword.py find_keywords 00 great 
"""

if __name__ == '__main__':
    base_dir = "/n/fs/nlp-mengzhou/space3/meta/data/sample_roberta"
    output_base_dir = "/n/fs/nlp-mengzhou/space3/meta/data/created" 

    function = sys.argv[1]
    index = sys.argv[2]
    keyword = sys.argv[3]
    templates_names = None if len(sys.argv) <= 4 else sys.argv[4]
    object = None if len(sys.argv) <= 4 else sys.argv[5]

    file = f"{base_dir}/train.10%.{index}"
    cache_dir = "/n/fs/nlp-mengzhou/space3/.cache/huggingface"
    keyword_dataset_dir = os.path.join(output_base_dir, "heuristic", "keywords-v2", "keyword_dataset")
    if object is not None:
        aggregated_dataset_dir = os.path.join(output_base_dir, "heuristic", "keywords-v2", "prompt_dataset", object)

    word_mapping = get_words_mapping()
    if function == "find_keywords":
        step1(file, cache_dir, keyword_dataset_dir)
    elif function == "find_keywords_aggregate":
        step2()
    elif function == "aggregate_and_masking":
        step3()

    elif function == "test":
        # text = "I have a good deal to do tonight. It was great."
        # # text = "It's a great evening!"
        # sent, char_start, char_end, keyword = find_keyword(text, keyword_in_context="great")
        # print(sent)
        # print(sent[char_start[0]: char_end[0]])
        #
        # print(filter_keyword(text, "great."))
        # print(filter_keyword(text, "great"))
        # print(filter_keyword(text, " great "))
        #
        # new_char_starts, new_char_ends, new_sents, new_keywords, new_labels = unify_final_words(text, ["great"], char_start, char_end, final_words_to_keywords=sst2_template()[1], obj="None")
        # for i, (s, e, sent) in enumerate(zip(new_char_starts, new_char_ends, new_sents)):
        #     print(sent)
        #     print(sent[s[0]: e[0]])
        #     print(new_labels[i])
        step3(test=True)
