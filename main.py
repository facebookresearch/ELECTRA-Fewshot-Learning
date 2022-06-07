#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import pdb
import random
import sys
from dataclasses import dataclass, field
from turtle import pd
from typing import Optional
from datasets import load_from_disk, load_metric, load_dataset

import utils
from templates import *
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
    set_seed,
    AutoModelForSequenceClassification
)
from transformers.models.auto import AutoModelForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from additional_args import AdditionalArguments
from utils import fix_config, get_random_subset, fix_datasets, read_tsv_file, convert_tsv_data_to_dict, tokenize_and_mapping, sanity_check_for_texts, downstream_tasks, numpy_seed, load_from_jsonl
from models import * 
from fewshot_trainer import FewShotTrainer
from sklearn.metrics import confusion_matrix
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.13.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "boolq": ("passage", "question")
}

mlm_key = "prompt"

task_to_template = {
    "rte": {0: PETStyleTwoWayNLITemplate, 1: PETStyleTwoWayNLIV2Template, 2: PETStyleTwoWayNLIV3Template},
    "qnli": PETStyleTwoWayNLITemplate,
    "sst2": PETStyleSST2Template,
    "sst5": PETStyleSST5Template,
    "mr": PETStyleMRTemplate,
    "yelp_polarity": PETStyleYelpTemplate,
    "mnli": {0: PETStyleMNLITemplate, 1: PETStyleMNLIV2Template, 2: PETStyleMNLIV3Template},
    "mnli-mm": PETStyleMNLITemplate,
    "yelp_full": PETStyleFullYelpTemplate,
    "snli": PETStyleSNLITemplate,
    "boolq": PETStyleBoolQTemplate,
    "ag_news": PETStyleAGNewsV4Template,
    "imdb": PETStyleYelpTemplate,
    "sst2_aug": PETStyleSST2Template,
    "sst2_v2_aug": PETStyleSST2Template,
    "sst5_aug": PETStyleSST5Template,
    "agnews_capital_aug": PETStyleAGNewsV4Template,
    "nli_aug": PETStyleMNLITemplate,
    "nli_v2_aug": PETStyleMNLITemplate,
    "boolq_aug": PETStyleBoolQTemplate,
    "copa": {0: PETCOPATemplateV1, 1: PETCOPATemplateV2, 2: COPATemplate},
    "copa-v3": COPATemplate,
    "copa-retrieval0.15": COPATemplate,
    "storycloze": {2: StoryClozeTemplate, 1: PETStoryClozeV2Template, 0: PETStoryClozeV1Template},
    "hellaswag": {2: HellaswagTemplate, 1: PETHellaswagV2Template},
    "piqa": {1: PETPIQAV2Template, 2: PIQATemplate},
    "trec": None
}

task_to_metric = {"rte": "accuracy",
                  "qnli": "accuracy",
                  "sst2": "accuracy",
                  "sst5": "accuracy",
                  "mr": "accuracy",
                  "yelp_polarity": "accuracy",
                  "mnli": "accuracy",
                  "mnli-mm": "accuracy",
                  "yelp_full": "accuracy",
                  "snli": "accuracy",
                  "boolq": "accuracy",
                  "ag_news": "accuracy",
                  "imdb": "accuracy",
                  "sst2_aug": "accuracy",
                  "sst2_v2_aug": "accuracy",
                  "sst5_aug": "accuracy",
                  "agnews_capital_aug": "accuracy",
                  "nli_aug": "accuracy",
                  "nli_v2_aug": "accuracy",
                  "boolq_aug": "accuracy",
                  "copa-v3": "accuracy",
                  "copa-v2": "accuracy",
                  "copa": "accuracy",
                  "copa-retrieval0.15": "accuracy",
                  "storycloze": "accuracy",
                  "hellaswag": "accuracy",
                  "piqa": "accuracy",
                  "trec": "accuracy"}

task_num_labels = {"mnli": 3, "mnli-mm": 3, "ag_news": 4, "sst5_aug": 5, "snli":3, "sst5": 5, "agnews_capital_aug": 4, "nli_aug": 3, "hellaswag": 4, "nli_v2_aug": 3}
span_level_tasks = ["hellaswag", "copa", "storycloze", "piqa", "wsc"]
logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            # commented out
            # assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."
        
        if self.dataset_config_name == "None":
            self.dataset_config_name = None
        


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdditionalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, additional_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, additional_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Additional parameters {additional_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
        t_name = data_args.task_name
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
        if data_args.dataset_config_name is not None:
            t_name = data_args.dataset_config_name
        else:
            t_name = data_args.dataset_name
    else:
        t_name = additional_args.add_task_name
        
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        if t_name in span_level_tasks:
            data_dir = os.path.dirname(data_args.train_file) 
            extension = os.path.basename(data_args.train_file).split(".")[1]
            # train: len(label) * k examples
            # validation: len(label) * k examples
            # fewshot_validation: test set for few-shot training  
            data_files = {"train": data_args.train_file,
                          "validation": data_args.train_file.replace("train", "val"), 
                          "fewshot_validation": os.path.join(data_dir, f"val.{extension}")}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
        elif data_args.train_file.endswith("json"):
            # Loading a dataset from local json files
            raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
        elif data_args.train_file.endswith("pt"):
            raw_datasets = datasets.DatasetDict()
            for split in data_files:
                data = load_from_disk(data_files[split])
                raw_datasets[split] = data
        elif data_args.train_file.endswith("jsonl"):
            raw_datasets = datasets.DatasetDict()
            for split in data_files:
                data = convert_tsv_data_to_dict(load_from_jsonl(data_files[split]))
                data = datasets.Dataset.from_dict(data)
                raw_datasets[split] = data
        else:
            raw_datasets = datasets.DatasetDict()
            for split in data_files:
                data = convert_tsv_data_to_dict(read_tsv_file(data_files[split]))
                data = datasets.Dataset.from_dict(data)
                raw_datasets[split] = data
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    raw_datasets = fix_datasets(t_name, raw_datasets, additional_args)


    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    fix_config(config, additional_args)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    def fix_small(examples):
        lens = len(examples["prompt"])
        new_prompts = []
        for i in range(lens):
            prompt = examples["prompt"][i]
            prompt = prompt.replace("<mask>", tokenizer.mask_token)
            new_prompts.append(prompt)
        examples["prompt"] = new_prompts
        return examples

    if additional_args.FT_method == "prompt_training":
        raw_datasets = raw_datasets.map(fix_small, batched=True)

    if additional_args.FT_method == "sequence_classification":
        Model = AutoModelForSequenceClassification
    elif additional_args.FT_method == "linear_fit":
        Model = ElectraForSequenceClassificationStandardAnalysis
    elif additional_args.FT_method == "linear_prob":
        if "electra" in model_args.model_name_or_path:
            Model = ElectraForLinearProb
        else:
            Model = BertForLinearProb 
    elif additional_args.FT_method == "prompt_tuning" or additional_args.FT_method == "prompt_training":
        if additional_args.objective == "softmax":
            if "roberta" in model_args.model_name_or_path:
                Model = RobertaForPromptSoftmaxTuning
            elif "bert" in model_args.model_name_or_path:
                Model = BertForPromptSoftmaxTuning
            elif "electra" in model_args.model_name_or_path:
                Model = ElectraForPromptSoftmaxTuning
        else:
            if "roberta" in model_args.model_name_or_path:
                Model = RobertaForPromptTuning
            elif "bert" in model_args.model_name_or_path:
                Model = BertForPromptTuning
            elif "electra" in model_args.model_name_or_path:
                if additional_args.objective == "dis":
                    Model = ElectraForPromptTuning
                elif additional_args.objective == "parallel_dis":
                    Model = ElectraV2ForPromptTuning
                elif additional_args.objective == "mlm":
                    Model = ElectraGeneratorForPromptTuning
    elif additional_args.FT_method == "prompt_span_tuning" or additional_args.FT_method == "prompt_span_training":
        if "electra" in model_args.model_name_or_path:
            Model = ElectraForPromptSpanTraining 
        elif "roberta" in model_args.model_name_or_path:
            Model = RobertaForPromptSpanTraining
    else:
        print(f"{additional_args.FT_method} is not supported")
        sys.exit()

    model = Model.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None)
    logger.info(model)

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if "label" not in name]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        elif "premise" in non_label_column_names and "hypothesis" in non_label_column_names:
            sentence1_key, sentence2_key = "premise", "hypothesis"
        elif "passage" in non_label_column_names and "question" in non_label_column_names:
            sentence1_key, sentence2_key = "passage", "question"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

        if "aug" in t_name:
            sentence1_key, sentence2_key = "prompt", None

        # could be better
        if t_name == "mr":
            sentence1_key, sentence2_key = "sentence", None

    if additional_args.FT_method.startswith("prompt"):
        tokenized_sentence1_key = mlm_key
        tokenized_sentence2_key = None
    else:
        tokenized_sentence1_key = sentence1_key
        tokenized_sentence2_key = sentence2_key

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id # trained
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        label_to_id = label_name_to_id
        # if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            # label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        # else:
        #     logger.warning(
        #         "Your model seems to have been trained with labels, but they don't match the dataset: ",
        #         f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
        #         "\nIgnoring the model labels as a result.",
        #     )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples, template=None, label_name="label"):
        # For glue, label is id
        # For other, label is label
        result = tokenize_and_mapping(examples, template, tokenizer, max_seq_length, additional_args.objective, tokenized_sentence1_key, tokenized_sentence2_key, padding, 2 if t_name not in task_num_labels else task_num_labels[t_name])

        if t_name not in span_level_tasks and t_name in downstream_tasks and additional_args.FT_method == "prompt_tuning" \
                and additional_args.objective != "mlm":
            label_words = []
            for i in range(len(result["input_ids"])):
                input_ids = result["input_ids"][i]
                start_token = result["token_start"][i]
                label_words.extend([input_ids[s] for s in start_token])
            if len(set(label_words)) != num_labels:
                pdb.set_trace()
            assert len(set(label_words)) == num_labels

        # Map labels to IDs (not necessary for GLUE tasks)
        if data_args.task_name is None and label_to_id is not None and "label" in examples:
            if not (additional_args.FT_method == "prompt_tuning" and "dis" in additional_args.objective): # label has been converted
                if not additional_args.FT_method == "prompt_training":
                    result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples[label_name]]
        return result


    with training_args.main_process_first(desc="dataset map pre-processing"):
        template = None
        mask_idx = tokenizer.encode(tokenizer.mask_token)[1]
        if additional_args.k_shot > 0 and t_name not in span_level_tasks:
            idx, train_d = get_random_subset(raw_datasets['train'], additional_args.k_shot, additional_args.data_seed)
            _, raw_datasets['validation'] = get_random_subset(raw_datasets['train'], additional_args.k_shot, additional_args.data_seed, exclude_idxes=idx)
            raw_datasets['train'] = train_d
            # sanity_check_for_texts(raw_datasets["train"], raw_datasets["validation"], sentence1_key, sentence2_key)
        elif additional_args.k_shot > 0:
            train_lens = len(raw_datasets["train"])
            with numpy_seed(additional_args.data_seed):
                order = np.random.permutation(train_lens)
            train_index = order[:additional_args.k_shot]
            valid_index = order[additional_args.k_shot:additional_args.k_shot * 2]
            print("Chosen train index:", train_index)
            print("Chosen valid index:", valid_index)
            raw_datasets["validation"] = raw_datasets["train"].select(valid_index)
            raw_datasets["train"] = raw_datasets["train"].select(train_index)
        template = task_to_template[t_name]
        if isinstance(template, dict):
            template = template[additional_args.template_id]
        if t_name in ["ag_news", "boolq", "mnli", "snli", "rte", "qnli"]:
            template = template(label_list, model.config.id2label, sentence1_key, sentence2_key,
                                model_args.model_name_or_path)
        else:
            if template is not None:
                template = template(label_list, model.config.id2label, sentence1_key, sentence2_key)

        logger.info(f"label_list: {label_list}")
        logger.info(f"id2label: {model.config.id2label}")
        logger.info(f"label2id: {model.config.label2id}")

        if t_name in downstream_tasks and (additional_args.FT_method == "prompt_tuning" or additional_args.FT_method == "prompt_span_tuning"):
            if additional_args.objective == "mlm":
                raw_datasets = template.encode(raw_datasets, tokenizer.mask_token)
            elif "dis" in additional_args.objective: 
                raw_datasets = template.encode_with_label_words(raw_datasets)
            elif additional_args.objective == "softmax":
                raw_datasets = template.encode_concate_options(raw_datasets)
            else:
                print(f"{additional_args.objective} is not supported.")
                sys.exit()

        if training_args.do_train:
            if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
                n = 2 
                if t_name in task_num_labels:
                    n = task_num_labels[t_name]
                num_exs = data_args.max_train_samples // n
                total_train_num_exs = len(raw_datasets["train"]) // n
                selected_idx = np.random.permutation(total_train_num_exs)[:num_exs]
                selected_idx = np.concatenate([selected_idx * n + i for i in range(n)])
                raw_datasets["train"] = raw_datasets["train"].select(selected_idx)
        elif not additional_args.do_analysis:
            raw_datasets.pop("train")

        raw_datasets = raw_datasets.map(lambda x:
            preprocess_function(x, template),
            batched=True,
            batch_size=900,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        if additional_args.FT_method.startswith("prompt") and additional_args.objective == "mlm":
            raw_datasets = raw_datasets.filter(lambda x: mask_idx in x["input_ids"], num_proc=20)

        for key in raw_datasets:
            print(f"{key}: {len(raw_datasets[key])} examples")

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]

    if training_args.do_eval or additional_args.do_zero_shot_eval or additional_args.do_analysis:
        if "validation" not in raw_datasets and "fewshot_validation" not in raw_datasets and "fewshot_validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        if "validation" in raw_datasets:
            eval_dataset = raw_datasets["validation"]
        else:
            eval_dataset = raw_datasets["fewshot_validation_matched" if data_args.task_name == "mnli" else "fewshot_validation"]
        # eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(eval_pred: EvalPrediction):
        preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=eval_pred.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - eval_pred.label_ids) ** 2).mean().item()}
        else:
            logger.info(confusion_matrix(preds, eval_pred.label_ids))
            return {"accuracy": (preds == eval_pred.label_ids).astype(np.float32).mean().item()}

    def compute_metrics_for_linear_probing(eval_pred: EvalPrediction) :
        preds = eval_pred.predictions
        labels = eval_pred.label_ids
        return {"rsquared": preds.sum() / labels.sum()}
        
    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    def compute_parallel_dis_metrics(eval_pred: EvalPrediction):
        n = 2
        if t_name in task_num_labels:
            n = task_num_labels[t_name]

        preds = eval_pred.predictions
        labels = eval_pred.label_ids
        binary_preds = (sigmoid(preds.reshape(-1)) > 0.5).astype(np.float32)
        binary_accuracy = (binary_preds == labels).astype(np.float32).mean().item()
        preds = np.argmax(preds, axis=1)
        labels = labels.reshape(-1, n)
        assert (labels.sum(-1) == n - 1).all()
        labels = np.where(labels == 0)[1]
        logger.info(confusion_matrix(preds, labels))
        return {"accuracy": (preds == labels).astype(np.float32).mean().item(), "binary_accuracy": binary_accuracy}
 
    def compute_dis_metrics(eval_pred: EvalPrediction):
        n = 2
        if t_name in task_num_labels:
            n = task_num_labels[t_name]

        preds = eval_pred.predictions
        labels = eval_pred.label_ids
        binary_preds = (sigmoid(preds) > 0.5).astype(np.float32)
        binary_accuracy = (binary_preds == labels).astype(np.float32).mean().item()

        preds = preds.reshape(-1, n)
        preds = np.argmin(preds, axis=1)
        labels = labels.reshape(-1, n)
        assert (labels.sum(-1) == n - 1).all()
        labels = np.where(labels == 0)[1]
        logger.info(confusion_matrix(preds, labels))
        return {"accuracy": (preds == labels).astype(np.float32).mean().item(), "binary_accuracy": binary_accuracy}
    
    def compute_nonsym_dis_metrics(eval_pred: EvalPrediction):
        preds = eval_pred.predictions
        labels = eval_pred.label_ids
        preds = preds > 0.5
        logger.info(confusion_matrix(preds, labels))
        return {"accuracy": (preds == labels).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    static_input = None
    if additional_args.FT_method.startswith("prompt"):
        if additional_args.objective == "mlm":
            static_input = utils.get_static_input(tokenizer, template, eval_dataset, max_seq_length, "mlm", 2  if t_name not in task_num_labels else task_num_labels[t_name])
        if "dis" in additional_args.objective:
            if t_name == "copa-retrieval0.15":
                compute_metrics = compute_nonsym_dis_metrics
            else:
                compute_metrics = compute_dis_metrics
        if "parallel_dis" in additional_args.objective:
            compute_metrics = compute_parallel_dis_metrics 
    if additional_args.FT_method == "linear_fit":
        compute_metrics = compute_metrics_for_linear_probing
    
    all_eval_dataset = {"validation": eval_dataset}
    if t_name in downstream_tasks: 
        if t_name == "mnli":
            all_eval_dataset.update({"fewshot_validation_matched": raw_datasets["fewshot_validation_matched"], 
                                    "fewshot_validation_mismatched": raw_datasets["fewshot_validation_mismatched"]})
        else:
            all_eval_dataset.update({"fewshot_validation": raw_datasets["fewshot_validation"]})
    
    if additional_args.do_analysis:
        all_eval_dataset["train"] = raw_datasets["train"]
    
    # so far data seems fine
    trainer = FewShotTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=all_eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        static_input=static_input,
        task_metric=task_to_metric[t_name],
        objective=additional_args.objective,
        additional_args=additional_args,
    )
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload
        if additional_args.FT_method in ["prompt_tuning", "sequence_classification", "prompt_span_tuning"]:
            os.remove(os.path.join(training_args.output_dir, "pytorch_model.bin"))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:

        logger.info("*** Evaluate ***")
        logger.info(f"Sample 0: {eval_dataset[0]}")
        logger.info(f"Sample 10: {eval_dataset[10]}")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        # tasks = [data_args.task_name]
        # eval_datasets = [eval_dataset]
        # if data_args.task_name == "mnli":
        #     tasks.append("mnli-mm")
        #     eval_datasets.append(raw_datasets["validation_mismatched"])
        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        print("*** Eval Results ***")
        for key in all_eval_dataset:
            print(f"{key}: {trainer.metric_tracker.all_scores[f'{key}_eval_accuracy']}")

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    if additional_args.do_zero_shot_eval:
        logger.info("*** Zero Shot Evaluate ***")
        logger.info(f"Sample 0: {eval_dataset[0]}")
        logger.info(f"Sample 10: {eval_dataset[10]}")
        # Loop to handle MNLI double evaluation (matched, mis-matched)

        metrics = trainer.zero_shot_evaluate()

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("zero shot eval", metrics)
        trainer.save_metrics("zero shot eval", metrics)

    if additional_args.do_analysis:
        logger.info("*** Analysis ***")
        logger.info(f"Sample 0: {eval_dataset[0]}")
        logger.info(f"Sample 10: {eval_dataset[10]}")

        if additional_args.analysis_version == 1:
            trainer.analyze_v1("Analysis V1")
        elif additional_args.analysis_version == 2: # for saving reps
            trainer.analyze_v2("Analysis V2")


    if training_args.do_train and training_args.do_eval:
        logger.info("*** Best Results ***")
        for key in all_eval_dataset: 
            logger.info(f"{key}: {trainer.metric_tracker.all_scores[f'{key}_eval_accuracy']}")


    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
