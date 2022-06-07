# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.
import pdb
from sqlite3 import complete_statement

import datasets
import numpy as np
from utils import turn_char_to_token_boundary

class FewShotTemplate:
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None):
        self.labels = labels
        self.id_to_labels = id_to_labels
        self.sentence1_key = sentence1_key
        self.sentence2_key = sentence2_key

    def get_features(self, dataset):
        if isinstance(dataset, datasets.DatasetDict):
            features = dataset[list(dataset.keys())[0]].features.keys()
        else:
            features = dataset.features.keys()
        features = list(features)
        return features

    # encode examples
    def encode(self, dataset, mask):
        encode_examples = getattr(self, "encode_separate_examples", None)
        if not callable(encode_examples):
            encode_examples = getattr(self, "encode_examples", None)
        assert callable(encode_examples)
        prompt_dataset = dataset.map(lambda x: encode_examples(x, mask), batched=True)
        return prompt_dataset

    def encode_concate_options(self, dataset):
        encode_examples = getattr(self, "encode_separate_concate_options_examples", None)
        if not callable(encode_examples):
            encode_examples = getattr(self, "encode_concate_options_examples", None)
        assert callable(encode_examples)
        prompt_dataset = dataset.map(lambda x: encode_examples(x), batched=True)
        return prompt_dataset

    # encode examples with label words
    def encode_with_label_words(self, dataset):
        features = self.get_features(dataset)
        encode_examples_with_labels = getattr(self, "encode_examples_with_label_words", None)
        if not callable(encode_examples_with_labels):
            encode_examples_with_labels = getattr(self, "encode_separate_examples_with_label_words", None)
            assert callable(encode_examples_with_labels)
            if "truncatable" in features:
                features.remove("truncatable")
                features.remove("complete")
            if "char_start" in features: # to accomodate augmented datasets
                features.remove("char_start")
                features.remove("char_end")
        features.remove("label")
        prompt_dataset = dataset.map(lambda x: encode_examples_with_labels(x), batched=True, remove_columns=features)
        return prompt_dataset

    # deprecated
    def encode_masked_prompt_with_label_words(self, prompt, target, mask):
        prompts_with_verbalizer = []
        targets = []
        for i in range(len(self.id_to_labels)):
            v = self.label_to_verbalizer[self.id_to_labels[i]]
            prompt_with_verbalizer = prompt.replace(mask, v)
            prompts_with_verbalizer.append(prompt_with_verbalizer)
            if target == i:
                targets.append(0)
            else:
                targets.append(1)
        return prompts_with_verbalizer, targets

    def combine(self, truncatable, complete, max_lens, padding, token_starts=None, token_ends=None):
        input_idss = []
        attention_maskss = []
        for i, (tr, co) in enumerate(zip(truncatable, complete)):
            truncatable, complete = self.truncate_truncatable_and_complete(tr, co, max_lens)
            input_ids = truncatable + complete
            assert len(input_ids) <= max_lens
            lenss = len(input_ids)
            input_ids += [padding] * (max_lens - lenss)
            attention_masks = np.zeros(max_lens)
            attention_masks[:lenss] = 1
            input_idss.append(input_ids)
            attention_maskss.append(attention_masks.astype(int).tolist())
            if token_starts is not None:
                len_truncatable = len(truncatable)
                token_starts[i] = [s + len_truncatable - 1 for s in token_starts[i]] # -2 remove eos and sos
                token_ends[i] = [s + len_truncatable - 1 for s in token_ends[i]]

        return input_idss, attention_maskss

    def truncate_truncatable_and_complete(self, truncatable, complete, max_lens):
        complete = complete[1:] # get rid of CLS
        lens = len(complete)
        truncatable = truncatable[:-1][: max_lens - lens]
        # assert len(combined) <= max_lens
        return truncatable, complete


class COPATemplate(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
    
    def get_conjunc(self, question):
        if question == "cause":
            return " because "
        elif question == "effect":
            return " so "
        
    def encode_examples_with_label_words(self, examples):
        char_starts = []
        char_ends = []
        prompts = []
        targets = []
        for i in range(len(examples["premise"])):
            premise = examples["premise"][i].strip().rstrip(".?!,;:'")
            question = examples["question"][i]
            label = examples["label"][i]
            choice1 = examples["choice1"][i]
            choice2 = examples["choice2"][i]
            conjunc = self.get_conjunc(question)
            prompt = premise
            char_start = len(prompt)

            for i, choice in enumerate([choice1, choice2]):
                choice = choice[0].lower() + choice[1:]
                complete_prompt = prompt + conjunc + choice
                char_end = len(complete_prompt)
                char_starts.append([char_start])
                char_ends.append([char_end])
                prompts.append(complete_prompt)
                targets.append(int(i != label))
        examples.update({"char_start": char_starts, "char_end": char_ends, "prompt": prompts, "label": targets})
        return examples

class PETCOPATemplateV1(COPATemplate):
    def encode_examples_with_label_words(self, examples):
        char_starts = []
        char_ends = []
        prompts = []
        targets = []
        for i in range(len(examples["premise"])):
            premise = examples["premise"][i].strip().rstrip(".?!,;:'")
            question = examples["question"][i]
            label = examples["label"][i]
            choice1 = examples["choice1"][i].rstrip(".?!,;:'")
            choice2 = examples["choice2"][i].rstrip(".?!,;:'")
            conjunc = self.get_conjunc(question)
            prompt = f'"{choice1}" or "{choice2}"? {premise} {conjunc} '
            char_start = len(prompt)

            for i, choice in enumerate([choice1, choice2]):
                choice = choice[0].lower() + choice[1:]
                complete_prompt = f'{prompt}{choice}.' 
                char_end = len(complete_prompt)
                char_starts.append([char_start])
                char_ends.append([char_end])
                prompts.append(complete_prompt)
                targets.append(int(i != label))
        examples.update({"char_start": char_starts, "char_end": char_ends, "prompt": prompts, "label": targets})
        return examples

class PETCOPATemplateV2(COPATemplate):
    def encode_examples_with_label_words(self, examples):
        char_starts = []
        char_ends = []
        prompts = []
        targets = []
        for i in range(len(examples["premise"])):
            premise = examples["premise"][i].strip().rstrip(".?!,;:'")
            question = examples["question"][i]
            label = examples["label"][i]
            choice1 = examples["choice1"][i].rstrip(".?!,;:'")
            choice2 = examples["choice2"][i].rstrip(".?!,;:'")
            conjunc = self.get_conjunc(question)
            prompt = f'{choice1} or {choice2}? {premise}'
            char_start = len(prompt)

            for i, choice in enumerate([choice1, choice2]):
                choice = choice[0].lower() + choice[1:]
                complete_prompt = prompt + conjunc + choice
                char_end = len(complete_prompt)
                char_starts.append([char_start])
                char_ends.append([char_end])
                complete_prompt += "."
                prompts.append(complete_prompt)
                targets.append(int(i != label))
        examples.update({"char_start": char_starts, "char_end": char_ends, "prompt": prompts, "label": targets})
        return examples

# class RecordTemplateV1(FewShotTemplate):
#     def encode_examples_with_label_words(self, examples):
#         char_starts = []
#         char_ends = []
#         prompts = []
#         targets = []
#         for i in range(len(examples["text"])):
#             text = examples["text"][i].strip()
#             pronoun = examples[""][i]
#             target = examples["span1_text"][i]

#             prompt = text + f" The pronoun *{pronoun}* refers to "
#             char_start = len(prompt)

#             for i, choice in enumerate([choice1, choice2]):
#                 choice = choice[0].lower() + choice[1:]
#                 complete_prompt = f'{prompt}{choice}' 
#                 char_end = len(complete_prompt)
#                 char_starts.append([char_start])
#                 char_ends.append([char_end])
#                 complete_prompt += "."
#                 prompts.append(complete_prompt)
#                 targets.append(int(i != label))
#         examples.update({"char_start": char_starts, "char_end": char_ends, "prompt": prompts, "label": targets})
#         return examples

    
class StoryClozeTemplate(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
 
    def encode_examples_with_label_words(self, examples):
        char_starts = []
        char_ends = []
        prompts = []
        targets = []
        for i in range(len(examples["InputSentence1"])):
            premise = " ".join([examples[f"InputSentence{j}"][i] for j in range(1, 5)])
            choice1 = examples["RandomFifthSentenceQuiz1"][i]
            choice2 = examples["RandomFifthSentenceQuiz2"][i]
            label = examples["label"][i]
            prompt = premise + " "
            char_start = len(prompt)

            for i, choice in enumerate([choice1, choice2]):
                complete_prompt = prompt + choice
                char_end = len(complete_prompt)
                char_starts.append([char_start])
                char_ends.append([char_end])
                prompts.append(complete_prompt)
                targets.append(int(i != label))
        examples.update({"char_start": char_starts, "char_end": char_ends, "prompt": prompts, "label": targets})
        return examples


class PETStoryClozeV1Template(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
 
    def encode_examples_with_label_words(self, examples):
        char_starts = []
        char_ends = []
        prompts = []
        targets = []
        for i in range(len(examples["InputSentence1"])):
            premise = " ".join([examples[f"InputSentence{j}"][i] for j in range(1, 5)])
            choice1 = examples["RandomFifthSentenceQuiz1"][i].rstrip(".?!,;:'")
            choice2 = examples["RandomFifthSentenceQuiz2"][i].rstrip(".?!,;:'")
            label = examples["label"][i]
            prompt = f'"{choice1}" or "{choice2}"? {premise} '
            char_start = len(prompt)

            for i, choice in enumerate([choice1, choice2]):
                complete_prompt = prompt + choice
                char_end = len(complete_prompt)
                char_starts.append([char_start])
                char_ends.append([char_end])
                complete_prompt += "."
                prompts.append(complete_prompt)
                targets.append(int(i != label))
        examples.update({"char_start": char_starts, "char_end": char_ends, "prompt": prompts, "label": targets})
        return examples

class PETStoryClozeV2Template(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
 
    def encode_examples_with_label_words(self, examples):
        char_starts = []
        char_ends = []
        prompts = []
        targets = []
        for i in range(len(examples["InputSentence1"])):
            premise = " ".join([examples[f"InputSentence{j}"][i] for j in range(1, 5)])
            choice1 = examples["RandomFifthSentenceQuiz1"][i].rstrip(".?!,;:'")
            choice2 = examples["RandomFifthSentenceQuiz2"][i].rstrip(".?!,;:'")
            label = examples["label"][i]
            prompt = f'{choice1} or {choice2}? {premise} '
            char_start = len(prompt)

            for i, choice in enumerate([choice1, choice2]):
                complete_prompt = prompt + choice
                char_end = len(complete_prompt)
                char_starts.append([char_start])
                char_ends.append([char_end])
                complete_prompt += "."
                prompts.append(complete_prompt)
                targets.append(int(i != label))
        examples.update({"char_start": char_starts, "char_end": char_ends, "prompt": prompts, "label": targets})
        return examples

class HellaswagTemplate(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
    
    def encode_examples_with_label_words(self, examples):
        char_starts = []
        char_ends = []
        prompts = []
        targets = []
        for i in range(len(examples["ctx"])):
            premise = examples["ctx"][i]
            label = examples["label"][i] 
            prompt = premise + " "
            char_start = len(prompt)

            for j in range(4):
                choice  = examples["endings"][i][j]
                complete_prompt = prompt + choice
                char_end = len(complete_prompt)
                char_starts.append([char_start])
                char_ends.append([char_end])
                prompts.append(complete_prompt)
                targets.append(int(j != label))
        examples.update({"char_start": char_starts, "char_end": char_ends, "prompt": prompts, "label": targets})
        return examples

class PETHellaswagV2Template(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
    
    def encode_examples_with_label_words(self, examples):
        char_starts = []
        char_ends = []
        prompts = []
        targets = []
        for i in range(len(examples["ctx"])):
            premise = examples["ctx"][i]
            label = examples["label"][i] 
            choices  = examples["endings"][i]
            choices = [choice.rstrip(".?!,;:'") for choice in choices]
            str_choices = " or ".join(choices)
            prompt = str_choices + f"? {premise} "
            char_start = len(prompt)

            for j in range(4):
                choice  = choices[j]
                complete_prompt = prompt + choice
                char_end = len(complete_prompt)
                char_starts.append([char_start])
                char_ends.append([char_end])
                complete_prompt += "."
                prompts.append(complete_prompt)
                targets.append(int(j != label))
        examples.update({"char_start": char_starts, "char_end": char_ends, "prompt": prompts, "label": targets})
        return examples

class PIQATemplate(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)

    def encode_separate_examples_with_label_words(self, examples):
        truncatables = []
        completes = []
        char_starts = []
        char_ends = []
        targets = []
        for i in range(len(examples["goal"])):
            premise = examples["goal"][i].strip()
            target = examples["label"][i]

            for j in range(2):
                choice  = examples[f"sol{j+1}"][i].strip()
                complete = " " + choice
                char_start = 1
                char_end = len(complete)
                char_starts.append([char_start])
                char_ends.append([char_end])
                truncatables.append(premise)
                completes.append(complete)
                targets.append(int(j != target))
        examples.update({"truncatable": truncatables, "complete": completes, "label": targets, "char_start": char_starts, "char_end": char_ends})
        return examples

    def encode_examples_with_label_words_dep(self, examples):
        char_starts = []
        char_ends = []
        prompts = []
        targets = []
        for i in range(len(examples["goal"])):
            premise = examples["goal"][i]
            label = examples["label"][i]
            prompt = premise + " "
            char_start = len(prompt)

            for j in range(2):
                choice  = examples[f"sol{j+1}"][i]
                complete_prompt = prompt + choice[0].lower() + choice[1:]
                char_end = len(complete_prompt)
                char_starts.append([char_start])
                char_ends.append([char_end])
                prompts.append(complete_prompt)
                targets.append(int(j != label))
        examples.update({"char_start": char_starts, "char_end": char_ends, "prompt": prompts, "label": targets})
        return examples


class PETPIQAV2Template(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)

    def encode_separate_examples_with_label_words(self, examples):
        truncatables = []
        completes = []
        char_starts = []
        char_ends = []
        targets = []
        for i in range(len(examples["goal"])):
            target = examples["label"][i]
            sol1 = examples["sol1"][i].strip().rstrip(".?!,;:'")
            sol2 = examples["sol2"][i].strip().rstrip(".?!,;:'")
            premise = examples["goal"][i].strip()
            prompt = f"{sol1} or {sol2}? {premise}"
            choices = [sol1, sol2]
            for j in range(2):
                choice = choices[j]
                complete = " " + choice
                char_start = 1
                char_end = len(complete)
                char_starts.append([char_start])
                char_ends.append([char_end])
                truncatables.append(prompt)
                complete += "."
                completes.append(complete)
                targets.append(int(j != target))
        examples.update({"truncatable": truncatables, "complete": completes, "label": targets, "char_start": char_starts, "char_end": char_ends})
        return examples

class PETStyleTwoWayNLIV2Template(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None, model_name=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
        if "0" in list(id_to_labels.values()):
            self.label_to_verbalizer = {'0': "Yes", "1": "No"}
        else:
            self.label_to_verbalizer = {"entailment": "Yes", "not_entailment": "No"}
        self.model_name = model_name

    # mask is different for different models
    def encode_examples(self, examples, mask):
        prompts = []
        for i in range(len(examples[self.sentence1_key])):
            sentence1 = examples[self.sentence1_key][i].rstrip(".?!,;:'")
            sentence2 = examples[self.sentence2_key][i]

            premise_strings = sentence1.rstrip(".?!,;:'")
            question_strings = sentence2
            prompts.append(f"{premise_strings}? {mask}. {question_strings}")
        examples.update({"prompt": prompts})
        return examples

    def encode_examples_with_label_words(self, examples):
        char_starts, char_ends = [], []
        prompts_with_verbalizer = []
        targets = []
        for i in range(len(examples[self.sentence1_key])):
            sentence1 = examples[self.sentence1_key][i].rstrip(".?!,;:'")
            sentence2 = examples[self.sentence2_key][i]

            premise_strings = sentence1.rstrip(".?!,;:'")
            question_strings = sentence2
            target = examples["label"][i]
            char_start = len(f"{premise_strings}? ")

            for j in range(len(self.id_to_labels)):
                v = self.label_to_verbalizer[self.id_to_labels[j]]
                prompt_with_verbalizer = f"{premise_strings}? {v}. {question_strings}"
                prompts_with_verbalizer.append(prompt_with_verbalizer)
                if target == j:
                    targets.append(0)
                else:
                    targets.append(1)
                char_end = len(f"{premise_strings}? {v}")
                char_starts.append([char_start])
                char_ends.append([char_end])
        return {"char_start": char_starts, "char_end": char_ends, "label": targets, "prompt": prompts_with_verbalizer}


class PETStyleTwoWayNLIV3Template(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None, model_name=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
        if "0" in list(id_to_labels.values()):
            self.label_to_verbalizer = {'0': "Yes", "1": "No"}
        else:
            self.label_to_verbalizer = {"entailment": "Yes", "not_entailment": "No"}
        self.model_name = model_name

    # mask is different for different models
    def encode_examples(self, examples, mask):
        prompts = []
        for i in range(len(examples[self.sentence1_key])):
            sentence1 = examples[self.sentence1_key][i].rstrip(".?!,;:'")
            sentence2 = examples[self.sentence2_key][i]

            premise_strings = sentence1.rstrip(".?!,;:'")
            question_strings = sentence2[0].lower() + sentence2[1:]
            prompts.append(f'"{premise_strings}"? {mask}, "{question_strings}"')
        examples.update({"prompt": prompts})
        return examples

    def encode_examples_with_label_words(self, examples):
        char_starts, char_ends = [], []
        prompts_with_verbalizer = []
        targets = []
        for i in range(len(examples[self.sentence1_key])):
            sentence1 = examples[self.sentence1_key][i].rstrip(".?!,;:'")
            sentence2 = examples[self.sentence2_key][i]

            premise_strings = sentence1.rstrip(".?!,;:'")
            question_strings = sentence2[0].lower() + sentence2[1:]
            target = examples["label"][i]
            char_start = len(f'"{premise_strings}"? ')

            for j in range(len(self.id_to_labels)):
                v = self.label_to_verbalizer[self.id_to_labels[j]]
                prompt_with_verbalizer = f'"{premise_strings}"? {v}, "{question_strings}"'
                prompts_with_verbalizer.append(prompt_with_verbalizer)
                if target == j:
                    targets.append(0)
                else:
                    targets.append(1)
                char_end = len(f'"{premise_strings}"? {v}')
                char_starts.append([char_start])
                char_ends.append([char_end])
        return {"char_start": char_starts, "char_end": char_ends, "label": targets, "prompt": prompts_with_verbalizer}

class PETStyleTwoWayNLITemplate(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None, model_name=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
        if "0" in list(id_to_labels.values()):
            self.label_to_verbalizer = {'0': "Yes", "1": "No"}
        else:
            self.label_to_verbalizer = {"entailment": "Yes", "not_entailment": "No"}
        self.model_name = model_name

    def encode_concate_options_examples(self, examples):
        char_starts, char_ends = [], []
        prompts_with_verbalizer = []
        for i in range(len(examples[self.sentence1_key])):
            cs, ce = [], []
            sentence1 = examples[self.sentence1_key][i].rstrip(".?!,;:'")
            sentence2 = examples[self.sentence2_key][i]

            premise_strings = sentence1.rstrip(".?!,;:'")
            question_strings = sentence2.lower()

            prompt_with_verbalizer = f"{premise_strings}? "

            for j in range(len(self.id_to_labels)):
                char_start = len(prompt_with_verbalizer)
                cs.append(char_start)

                v = self.label_to_verbalizer[self.id_to_labels[j]]
                prompt_with_verbalizer += f"{v}"
                char_end = len(prompt_with_verbalizer)
                ce.append(char_end)

                if "roberta" not in self.model_name:
                    prompt_with_verbalizer += " "

            prompt_with_verbalizer = prompt_with_verbalizer.strip() + f", {question_strings}"
            char_starts.append(cs)
            char_ends.append(ce)
            prompts_with_verbalizer.append(prompt_with_verbalizer)
        return {"char_start": char_starts, "char_end": char_ends, "prompt": prompts_with_verbalizer}

    # mask is different for different models
    def encode_examples(self, examples, mask):
        prompts = []
        for i in range(len(examples[self.sentence1_key])):
            sentence1 = examples[self.sentence1_key][i].rstrip(".?!,;:'")
            sentence2 = examples[self.sentence2_key][i]

            premise_strings = sentence1.rstrip(".?!,;:'")
            question_strings = sentence2.lower()
            prompts.append(f"{premise_strings}? {mask}, {question_strings}")
        examples.update({"prompt": prompts})
        return examples

    def encode_examples_with_label_words(self, examples):
        char_starts, char_ends = [], []
        prompts_with_verbalizer = []
        targets = []
        for i in range(len(examples[self.sentence1_key])):
            sentence1 = examples[self.sentence1_key][i].rstrip(".?!,;:'")
            sentence2 = examples[self.sentence2_key][i]

            premise_strings = sentence1.rstrip(".?!,;:'")
            question_strings = sentence2.lower()
            target = examples["label"][i]
            char_start = len(f"{premise_strings}? ")

            for j in range(len(self.id_to_labels)):
                v = self.label_to_verbalizer[self.id_to_labels[j]]
                prompt_with_verbalizer = f"{premise_strings}? {v}, {question_strings}"
                prompts_with_verbalizer.append(prompt_with_verbalizer)
                if target == j:
                    targets.append(0)
                else:
                    targets.append(1)
                char_end = len(f"{premise_strings}? {v}")
                char_starts.append([char_start])
                char_ends.append([char_end])
        return {"char_start": char_starts, "char_end": char_ends, "label": targets, "prompt": prompts_with_verbalizer}

class PETStyleAGNewsTemplate(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
        self.label_to_verbalizer = {0: "world", 1: "sports", 2: "business", 3: "technology"}

    # mask is different for different models
    def encode_examples(self, examples, mask):
        prompts = []
        for i in range(len(examples[self.sentence1_key])):
            sentence = examples[self.sentence1_key][i].strip()
            prompts.append(f"{sentence} The topic is on {mask}.")
        examples.update({"prompt": prompts})
        return examples

    def encode_examples_with_label_words(self, examples):
        char_starts, char_ends = [], []
        prompts_with_verbalizer = []
        targets = []
        for i in range(len(examples[self.sentence1_key])):

            sentence = examples[self.sentence1_key][i].strip()
            target = examples["label"][i]
            char_start = len(f"{sentence} The topic is on ")

            for j in range(len(self.id_to_labels)):
                v = self.label_to_verbalizer[self.id_to_labels[j]]
                prompt_with_verbalizer = f"{sentence} The topic is on {v}."
                prompts_with_verbalizer.append(prompt_with_verbalizer)
                if target == j:
                    targets.append(0)
                else:
                    targets.append(1)
                char_end = len(f"{sentence} The topic is on {v}")
                char_starts.append([char_start])
                char_ends.append([char_end])
        return {"char_start": char_starts, "char_end": char_ends, "label": targets, "prompt": prompts_with_verbalizer}


class PETStyleAGNewsV2Template(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
        self.label_to_verbalizer = {0: "world", 1: "sports", 2: "business", 3: "technology"}

    # mask is different for different models
    def encode_examples(self, examples, mask):
        prompts = []
        for i in range(len(examples[self.sentence1_key])):
            sentence = examples[self.sentence1_key][i].strip()
            prompts.append(f"{sentence} It is about {mask}.")
        examples.update({"prompt": prompts})
        return examples

    def encode_examples_with_label_words(self, examples):
        char_starts, char_ends = [], []
        prompts_with_verbalizer = []
        targets = []
        for i in range(len(examples[self.sentence1_key])):

            sentence = examples[self.sentence1_key][i].strip()
            target = examples["label"][i]
            char_start = len(f"{sentence} It is about ")

            for j in range(len(self.id_to_labels)):
                v = self.label_to_verbalizer[self.id_to_labels[j]]
                prompt_with_verbalizer = f"{sentence} It is about {v}."
                prompts_with_verbalizer.append(prompt_with_verbalizer)
                if target == j:
                    targets.append(0)
                else:
                    targets.append(1)
                char_end = len(f"{sentence} It is about {v}")
                char_starts.append([char_start])
                char_ends.append([char_end])
        return {"char_start": char_starts, "char_end": char_ends, "label": targets, "prompt": prompts_with_verbalizer}


class PETStyleAGNewsV3Template(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
        self.label_to_verbalizer = {0: "World", 1: "Sports", 2: "Business", 3: "Tech"}

    # mask is different for different models
    def encode_examples(self, examples, mask):
        prompts = []
        for i in range(len(examples[self.sentence1_key])):
            sentence = examples[self.sentence1_key][i].strip()
            prompts.append(f"{mask}: {sentence}")
        examples.update({"prompt": prompts})
        return examples

    def encode_examples_with_label_words(self, examples):
        char_starts, char_ends = [], []
        prompts_with_verbalizer = []
        targets = []
        for i in range(len(examples[self.sentence1_key])):

            sentence = examples[self.sentence1_key][i].strip()
            target = examples["label"][i]
            char_start = 0

            for j in range(len(self.id_to_labels)):
                v = self.label_to_verbalizer[self.id_to_labels[j]]
                prompt_with_verbalizer = f"{v}: {sentence}"
                prompts_with_verbalizer.append(prompt_with_verbalizer)
                if target == j:
                    targets.append(0)
                else:
                    targets.append(1)
                char_end = len(f"{v}")
                char_starts.append([char_start])
                char_ends.append([char_end])
        return {"char_start": char_starts, "char_end": char_ends, "label": targets, "prompt": prompts_with_verbalizer}


class PETStyleAGNewsV4Template(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None, model_name=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
        self.label_to_verbalizer = {0: "World", 1: "Sports", 2: "Business", 3: "Tech"}
        self.model_name = model_name

    # mask is different for different models
    def encode_examples(self, examples, mask):
        prompts = []
        for i in range(len(examples[self.sentence1_key])):
            sentence = examples[self.sentence1_key][i].strip()
            prompts.append(f"{mask}: {sentence}")
        examples.update({"prompt": prompts})
        return examples

    def encode_examples_with_label_words(self, examples):
        char_starts, char_ends = [], []
        prompts_with_verbalizer = []
        targets = []
        for i in range(len(examples[self.sentence1_key])):

            sentence = examples[self.sentence1_key][i].strip()
            target = examples["label"][i]
            char_start = 0

            for j in range(len(self.id_to_labels)):
                v = self.label_to_verbalizer[self.id_to_labels[j]]
                prompt_with_verbalizer = f"{v} News: {sentence}"
                prompts_with_verbalizer.append(prompt_with_verbalizer)
                if target == j:
                    targets.append(0)
                else:
                    targets.append(1)
                char_end = len(f"{v}")
                char_starts.append([char_start])
                char_ends.append([char_end])
        return {"char_start": char_starts, "char_end": char_ends, "label": targets, "prompt": prompts_with_verbalizer}

    def encode_concate_options_examples(self, examples):
        char_starts, char_ends = [], []
        prompts_with_verbalizer = []
        for i in range(len(examples[self.sentence1_key])):
            cs, ce = [], []
            sentence = examples[self.sentence1_key][i].strip()
            prompt_with_verbalizer = ""
            for j in range(len(self.id_to_labels)):
                v = self.label_to_verbalizer[self.id_to_labels[j]]
                char_start = len(prompt_with_verbalizer)
                prompt_with_verbalizer += f"{v}"
                char_end = len(prompt_with_verbalizer)
                cs.append(char_start)
                ce.append(char_end)
                if "roberta" not in self.model_name:
                    prompt_with_verbalizer += " "
            char_starts.append(cs)
            char_ends.append(ce)
            prompt_with_verbalizer = prompt_with_verbalizer.strip() + f" News: {sentence}"
            prompts_with_verbalizer.append(prompt_with_verbalizer)
        return {"char_start": char_starts, "char_end": char_ends, "prompt": prompts_with_verbalizer}



class PETStyleSST2Template(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
        self.id_to_labels = {0: "negative", 1: "positive"}
        self.label_to_verbalizer = {"positive": "great", "negative": "terrible"}


    def encode_concate_options_examples(self, examples):
        char_starts, char_ends = [], []
        prompts_with_verbalizer = []
        for i in range(len(examples[self.sentence1_key])):
            cs, ce = [], []
            sentence = examples[self.sentence1_key][i].strip()

            prompt_with_verbalizer = f"{sentence} It was "

            for j in range(len(self.id_to_labels)):
                char_start = len(prompt_with_verbalizer)
                cs.append(char_start)

                v = self.label_to_verbalizer[self.id_to_labels[j]]
                prompt_with_verbalizer += f"{v}"
                char_end = len(prompt_with_verbalizer)
                ce.append(char_end)

                prompt_with_verbalizer += " "

            prompt_with_verbalizer = prompt_with_verbalizer.strip() + "."
            char_starts.append(cs)
            char_ends.append(ce)
            prompts_with_verbalizer.append(prompt_with_verbalizer)
        return {"char_start": char_starts, "char_end": char_ends, "prompt": prompts_with_verbalizer}

    # mask is different for different models
    def encode_examples(self, examples, mask):
        prompts = []
        for i in range(len(examples[self.sentence1_key])):
            sentence = examples[self.sentence1_key][i].strip()
            prompts.append(f"{sentence} It was {mask}.")
        examples.update({"prompt": prompts})
        return examples

    def encode_examples_with_label_words(self, examples):
        char_starts, char_ends = [], []
        prompts_with_verbalizer = []
        targets = []
        for i in range(len(examples[self.sentence1_key])):
            sentence = examples[self.sentence1_key][i].strip()
            target = examples["label"][i]
            char_start = len(f"{sentence} It was ")

            for j in range(len(self.id_to_labels)):
                v = self.label_to_verbalizer[self.id_to_labels[j]]
                prompt_with_verbalizer = f"{sentence} It was {v}."
                prompts_with_verbalizer.append(prompt_with_verbalizer)
                if target == j:
                    targets.append(0)
                else:
                    targets.append(1)
                char_end = len(f"{sentence} It was {v}")
                char_starts.append([char_start])
                char_ends.append([char_end])
        return {"char_start": char_starts, "char_end": char_ends, "label": targets, "prompt": prompts_with_verbalizer}


class PETStyleYelpTemplate(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
        self.label_to_verbalizer = {1: "great", 0: "terrible"}

    def encode_separate_concate_options_examples(self, examples):
        truncatable = []
        complete = []
        char_starts = []
        char_ends = []
        for i in range(len(examples[self.sentence1_key])):
            sentence = examples[self.sentence1_key][i]
            truncatable.append(f"{sentence}")
            complete_prompt = f" It was "
            cs, ce = [], []
            for j in range(len(self.id_to_labels)):
                v = self.label_to_verbalizer[self.id_to_labels[j]]
                char_start = len(complete_prompt)
                complete_prompt += v
                char_end = len(complete_prompt)

                cs.append(char_start)
                ce.append(char_end)
                complete_prompt += " "
            complete_prompt = complete_prompt.strip() + "."
            complete.append(complete_prompt)
            char_starts.append(cs)
            char_ends.append(ce)
        examples.update({"truncatable": truncatable, "complete": complete, "char_start": char_starts, "char_end": char_ends})
        return examples

    def encode_separate_examples_with_label_words(self, examples):
        truncatable = []
        complete = []
        char_starts = []
        char_ends = []
        targets = []
        for i in range(len(examples[self.sentence1_key])):
            sentence = examples[self.sentence1_key][i]
            target = examples["label"][i]
            for j in range(len(self.id_to_labels)):
                v = self.label_to_verbalizer[self.id_to_labels[j]]
                truncatable.append(f"{sentence}")
                complete.append(f" It was {v}.")
                if target == j:
                    targets.append(0)
                else:
                    targets.append(1)
                char_start = len(f" It was ")
                char_end = len(f" It was {v}")
                char_starts.append([char_start])
                char_ends.append([char_end])
        examples.update({"truncatable": truncatable, "complete": complete, "label": targets, "char_start": char_starts, "char_end": char_ends})
        return examples

    def encode_separate_examples(self, examples, mask):
        truncatable = []
        complete = []
        for i in range(len(examples[self.sentence1_key])):
            sentence = examples[self.sentence1_key][i]
            truncatable.append(f"{sentence}")
            complete.append(f" It was {mask}.")
        examples.update({"truncatable": truncatable})
        examples.update({"complete": complete})
        return examples

    # mask is different for different models
    def encode_examples(self, examples, mask):
        prompts = []
        for i in range(len(examples[self.sentence1_key])):
            sentence = examples[self.sentence1_key][i]
            prompts.append(f"{sentence} It was {mask}.")
        examples.update({"prompt": prompts})
        return examples



class PETStyleFullYelpTemplate(PETStyleYelpTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
        self.label_to_verbalizer = {1: "terrible", 2: "bad", 3: "okay", 4: "good", 5: "great"}

    def encode_separate_examples_with_label_words(self, examples):
        return super(PETStyleFullYelpTemplate, self).encode_separate_examples_with_label_words(examples)


class PETStyleSST5Template(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
        # "very positive, positive, neutral, negative, very negative"
        self.label_to_verbalizer = {0: "terrible", 1: "bad", 2: "okay", 3: "good", 4: "great"}


    # mask is different for different models
    def encode_examples(self, examples, mask):
        prompts = []
        for i in range(len(examples[self.sentence1_key])):
            sentence = examples[self.sentence1_key][i]
            prompts.append(f"{sentence} It was {mask}.")
        examples.update({"prompt": prompts})
        return examples

    def encode_examples_with_label_words(self, examples):
        char_starts, char_ends = [], []
        prompts_with_verbalizer = []
        targets = []
        for i in range(len(examples[self.sentence1_key])):
            sentence = examples[self.sentence1_key][i]
            target = examples["label"][i]
            char_start = len(f"{sentence} It was ")

            for j in range(len(self.id_to_labels)):
                v = self.label_to_verbalizer[self.id_to_labels[j]]
                prompt_with_verbalizer = f"{sentence} It was {v}."
                prompts_with_verbalizer.append(prompt_with_verbalizer)
                if target == self.id_to_labels[j]:
                    targets.append(0)
                else:
                    targets.append(1)
                char_end = len(f"{sentence} It was {v}")
                char_starts.append([char_start])
                char_ends.append([char_end])
        return {"char_start": char_starts, "char_end": char_ends, "label": targets, "prompt": prompts_with_verbalizer}

    # same as SST2
    def encode_concate_options_examples(self, examples):
        char_starts, char_ends = [], []
        prompts_with_verbalizer = []
        for i in range(len(examples[self.sentence1_key])):
            cs, ce = [], []
            sentence = examples[self.sentence1_key][i].strip()

            prompt_with_verbalizer = f"{sentence} It was "

            for j in range(len(self.id_to_labels)):
                char_start = len(prompt_with_verbalizer)
                cs.append(char_start)

                v = self.label_to_verbalizer[self.id_to_labels[j]]
                prompt_with_verbalizer += f"{v}"
                char_end = len(prompt_with_verbalizer)
                ce.append(char_end)

                prompt_with_verbalizer += " "

            prompt_with_verbalizer = prompt_with_verbalizer.strip() + "."
            char_starts.append(cs)
            char_ends.append(ce)
            prompts_with_verbalizer.append(prompt_with_verbalizer)
        return {"char_start": char_starts, "char_end": char_ends, "prompt": prompts_with_verbalizer}



class PETStyleMRTemplate(PETStyleSST2Template):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
        self.id_to_labels = id_to_labels
        self.label_to_verbalizer = {"0": "terrible", "1": "great"}

    def encode_concate_options_examples(self, examples):
        return super(PETStyleMRTemplate, self).encode_concate_options_examples(examples)

    def encode_examples_with_label_words(self, examples):
        char_starts, char_ends = [], []
        prompts_with_verbalizer = []
        targets = []
        for i in range(len(examples["sentence"])):

            sentence = examples["sentence"][i].strip()
            target = examples["label"][i]
            char_start = len(f"{sentence} It was ")

            for j in range(len(self.id_to_labels)):
                v = self.label_to_verbalizer[self.id_to_labels[j]]
                prompt_with_verbalizer = f"{sentence} It was {v}."
                prompts_with_verbalizer.append(prompt_with_verbalizer)
                if target == self.id_to_labels[j]: # the only difference from SST2
                    targets.append(0)
                else:
                    targets.append(1)
                char_end = len(f"{sentence} It was {v}")
                char_starts.append([char_start])
                char_ends.append([char_end])
        return {"char_start": char_starts, "char_end": char_ends, "label": targets, "prompt": prompts_with_verbalizer}


class PETStyleMNLITemplate(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None, model_name=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
        if 1 in list(id_to_labels.values()):
            self.label_to_verbalizer = {0: "Yes", 1: "Maybe", 2: "No"} # for augmented training
        elif "1" in list(id_to_labels.values()):
            self.label_to_verbalizer = {"0": 'Yes', "1": "Maybe", "2": "No"}
        else:
            self.label_to_verbalizer = {"entailment": "Yes", "neutral": "Maybe", "contradiction": "No"}
        self.model_name = model_name
        

    def encode_concate_options_examples(self, examples):
        char_starts, char_ends = [], []
        prompts_with_verbalizer = []
        for i in range(len(examples[self.sentence1_key])):
            cs, ce = [], []
            sentence1 = examples[self.sentence1_key][i].strip().rstrip(".?!,;:'")
            sentence2 = examples[self.sentence2_key][i].strip() if self.sentence2_key is not None else ""

            premise_strings = sentence1.rstrip(".?!,;:'")
            question_strings = sentence2.lower()

            prompt_with_verbalizer = f"{premise_strings}? "

            for j in range(len(self.id_to_labels)):
                char_start = len(prompt_with_verbalizer)
                cs.append(char_start)

                v = self.label_to_verbalizer[self.id_to_labels[j]]
                prompt_with_verbalizer += f"{v}"
                char_end = len(prompt_with_verbalizer)
                ce.append(char_end)

                if "roberta" not in self.model_name:
                    prompt_with_verbalizer += " "

            prompt_with_verbalizer = prompt_with_verbalizer.strip() + f", {question_strings}"
            char_starts.append(cs)
            char_ends.append(ce)
            prompts_with_verbalizer.append(prompt_with_verbalizer)
        return {"char_start": char_starts, "char_end": char_ends, "prompt": prompts_with_verbalizer}

    # mask is different for different models
    def encode_examples(self, examples, mask):
        prompts = []
        for i in range(len(examples[self.sentence1_key])):
            sentence1 = examples[self.sentence1_key][i].strip().rstrip(".?!,;:'")
            sentence2 = examples[self.sentence2_key][i].strip() if self.sentence2_key is not None else ""

            premise_strings = sentence1.rstrip(".?!,;:'")
            question_strings = sentence2[0].lower() + sentence2[1:]
            prompts.append(f"{premise_strings}? {mask}, {question_strings}")
        examples.update({"prompt": prompts})
        return examples

    def encode_examples_with_label_words(self, examples):
        char_starts, char_ends = [], []
        prompts_with_verbalizer = []
        targets = []
        for i in range(len(examples[self.sentence1_key])):

            sentence1 = examples[self.sentence1_key][i].strip().rstrip(".?!,;:'")
            sentence2 = examples[self.sentence2_key][i].strip() if self.sentence2_key is not None else ""
            sentence2 = sentence2[0].lower() + sentence2[1:]
            target = examples["label"][i]
            char_start = len(f"{sentence1}? ")

            for j in range(len(self.id_to_labels)):
                v = self.label_to_verbalizer[self.id_to_labels[j]]
                prompt_with_verbalizer = f"{sentence1}? {v}, {sentence2}."
                prompts_with_verbalizer.append(prompt_with_verbalizer)
                if target == j:
                    targets.append(0)
                else:
                    targets.append(1)
                char_end = len(f"{sentence1}? {v}")
                char_starts.append([char_start])
                char_ends.append([char_end])
        return {"char_start": char_starts, "char_end": char_ends, "label": targets, "prompt": prompts_with_verbalizer}


class PETStyleMNLIV2Template(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None, model_name=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
        if 1 in list(id_to_labels.values()):
            self.label_to_verbalizer = {0: "Yes", 1: "Maybe", 2: "No"} # for augmented training
        elif "1" in list(id_to_labels.values()):
            self.label_to_verbalizer = {"0": 'Yes', "1": "Maybe", "2": "No"}
        else:
            self.label_to_verbalizer = {"entailment": "Yes", "neutral": "Maybe", "contradiction": "No"}
        self.model_name = model_name
 
    # mask is different for different models
    def encode_examples(self, examples, mask):
        prompts = []
        for i in range(len(examples[self.sentence1_key])):
            sentence1 = examples[self.sentence1_key][i].strip().rstrip(".?!,;:'")
            sentence2 = examples[self.sentence2_key][i].strip() if self.sentence2_key is not None else ""

            premise_strings = sentence1.rstrip(".?!,;:'")
            question_strings = sentence2
            prompts.append(f"{premise_strings}? {mask}. {question_strings}")
        examples.update({"prompt": prompts})
        return examples

    def encode_examples_with_label_words(self, examples):
        char_starts, char_ends = [], []
        prompts_with_verbalizer = []
        targets = []
        for i in range(len(examples[self.sentence1_key])):

            sentence1 = examples[self.sentence1_key][i].strip().rstrip(".?!,;:'")
            sentence2 = examples[self.sentence2_key][i].strip() if self.sentence2_key is not None else ""
            target = examples["label"][i]
            char_start = len(f"{sentence1}? ")

            for j in range(len(self.id_to_labels)):
                v = self.label_to_verbalizer[self.id_to_labels[j]]
                prompt_with_verbalizer = f"{sentence1}? {v}. {sentence2}."
                prompts_with_verbalizer.append(prompt_with_verbalizer)
                if target == j:
                    targets.append(0)
                else:
                    targets.append(1)
                char_end = len(f"{sentence1}? {v}")
                char_starts.append([char_start])
                char_ends.append([char_end])
        return {"char_start": char_starts, "char_end": char_ends, "label": targets, "prompt": prompts_with_verbalizer}


class PETStyleMNLIV3Template(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None, model_name=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
        if 1 in list(id_to_labels.values()):
            self.label_to_verbalizer = {0: "Yes", 1: "Maybe", 2: "No"} # for augmented training
        elif "1" in list(id_to_labels.values()):
            self.label_to_verbalizer = {"0": 'Yes', "1": "Maybe", "2": "No"}
        else:
            self.label_to_verbalizer = {"entailment": "Yes", "neutral": "Maybe", "contradiction": "No"}
        self.model_name = model_name
 
    # mask is different for different models
    def encode_examples(self, examples, mask):
        prompts = []
        for i in range(len(examples[self.sentence1_key])):
            sentence1 = examples[self.sentence1_key][i].strip().rstrip(".?!,;:'")
            sentence2 = examples[self.sentence2_key][i].strip() if self.sentence2_key is not None else ""

            premise_strings = sentence1.rstrip(".?!,;:'")
            question_strings = sentence2[0].lower() + sentence2[1:]
            prompts.append(f'"{premise_strings}"? {mask}, "{question_strings}"')
        examples.update({"prompt": prompts})
        return examples

    def encode_examples_with_label_words(self, examples):
        char_starts, char_ends = [], []
        prompts_with_verbalizer = []
        targets = []
        for i in range(len(examples[self.sentence1_key])):

            sentence1 = examples[self.sentence1_key][i].strip().rstrip(".?!,;:'")
            sentence2 = examples[self.sentence2_key][i].strip() if self.sentence2_key is not None else ""
            sentence2 = sentence2[0].lower() + sentence2[1:]
            target = examples["label"][i]
            char_start = len(f'"{sentence1}"? ')

            for j in range(len(self.id_to_labels)):
                v = self.label_to_verbalizer[self.id_to_labels[j]]
                prompt_with_verbalizer = f'"{sentence1}"? {v}, "{sentence2}".'
                prompts_with_verbalizer.append(prompt_with_verbalizer)
                if target == j:
                    targets.append(0)
                else:
                    targets.append(1)
                char_end = len(f'"{sentence1}"? {v}')
                char_starts.append([char_start])
                char_ends.append([char_end])
        return {"char_start": char_starts, "char_end": char_ends, "label": targets, "prompt": prompts_with_verbalizer}

class PETStyleSNLITemplate(PETStyleMNLITemplate):
    def __init__(self, labels, id_to_labels, sentence1_key, sentence2_key=None, model_name=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key, model_name)
        self.label_to_verbalizer = {0: "Yes", 1: "Maybe", 2: "No"}

class PETStyleBoolQTemplate(FewShotTemplate):
    def __init__(self, labels, id_to_labels, sentence1_key=None, sentence2_key=None, model_name=None):
        super().__init__(labels, id_to_labels, sentence1_key, sentence2_key)
        self.label_to_verbalizer = {0: "No", 1: "Yes"}
        self.sentence1_key = "passage"
        self.sentence2_key = "question"
        if sentence1_key == "prompt":
            self.sentence1_key = "prompt"
            self.sentence2_key = None
        self.model_name = model_name

    def encode_separate_concate_options_examples(self, examples):
        truncatable = []
        complete = []
        char_starts = []
        char_ends = []
        for i in range(len(examples[self.sentence1_key])):
            cs = []
            ce = []
            passage = examples[self.sentence1_key][i]
            question = examples[self.sentence2_key][i]
            prompt_complete = f" Question: {question}? Answer: "
            truncatable.append(f"{passage}")
            for j in range(len(self.id_to_labels)):
                char_start = len(prompt_complete)
                v = self.label_to_verbalizer[self.id_to_labels[j]]
                prompt_complete += v
                char_end = len(prompt_complete)
                cs.append(char_start)
                ce.append(char_end)
                if "roberta" not in self.model_name:
                    prompt_complete += " "
            prompt_complete = prompt_complete.strip() + "."
            complete.append(prompt_complete)
            char_starts.append(cs)
            char_ends.append(ce)
        examples.update({"truncatable": truncatable, "complete": complete, "char_start": char_starts, "char_end": char_ends})
        return examples

    def encode_separate_examples_with_label_words(self, examples):
        truncatable = []
        complete = []
        char_starts = []
        char_ends = []
        targets = []
        for i in range(len(examples[self.sentence1_key])):
            passage = examples[self.sentence1_key][i]
            question = examples[self.sentence2_key][i] if self.sentence2_key is not None else ""

            target = examples["label"][i]
            for j in range(len(self.id_to_labels)):
                v = self.label_to_verbalizer[self.id_to_labels[j]]
                truncatable.append(f"{passage}")
                complete.append(f" Question: {question}? Answer: {v}.")
                if target == j:
                    targets.append(0)
                else:
                    targets.append(1)
                char_start = len(f" Question: {question}? Answer: ")
                char_end = len(f" Question: {question}? Answer: {v}")
                char_starts.append([char_start])
                char_ends.append([char_end])
        examples.update({"truncatable": truncatable, "complete": complete, "label": targets, "char_start": char_starts, "char_end": char_ends})
        return examples

    def encode_separate_examples(self, examples, mask):
        truncatable = []
        complete = []
        for i in range(len(examples[self.sentence1_key])):
            passage = examples[self.sentence1_key][i]
            question = examples[self.sentence2_key][i]
            truncatable.append(f"{passage}")
            complete.append(f" Question: {question}? Answer: {mask}")
        examples.update({"truncatable": truncatable})
        examples.update({"complete": complete})
        return examples

if __name__ == '__main__':
    " python3 ~/hf/templates.py "
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    raw_datasets = load_dataset("glue", "sst2")
    train_dataset = raw_datasets["train"]

    label_names = train_dataset.features["label"].names
    num_labels = len(label_names)
    label2id = {l: i for i, l in enumerate(label_names)}
    id2labels = {id: label for label, id in label2id.items()}
    template = PETStyleSST2Template(label_names, id2labels, "sentence")

    mask = tokenizer.encode(tokenizer.mask_token)[1]
    masked_prompt_dataset = template.encode(train_dataset, mask=tokenizer.mask_token)
    verbalized_prompt_dataset = template.encode_with_label_words(masked_prompt_dataset)

    sample_idx = [0, 4, 11]
    for i in sample_idx:
        re = tokenizer(verbalized_prompt_dataset[i]["prompt"], return_offsets_mapping=True)
        span_start, span_end = turn_char_to_token_boundary(re["offset_mapping"], verbalized_prompt_dataset[i]["char_start"], verbalized_prompt_dataset[i]["char_end"])
        for s, e in zip(span_start, span_end):
            tokens = re["input_ids"][s:e+1]
            print(tokens)
