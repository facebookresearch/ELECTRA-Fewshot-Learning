# Prompting ELECTRA: Few-Shot Learning with Discriminative Pre-Trained Models

This repository contains the code for our preprint paper [Prompting ELECTRA: Few-Shot Learning with Discriminative Pre-Trained Models](https://openreview.net/pdf?id=SdOLXkjaq19).

## Quick Links
- [Prompting ELECTRA: Few-Shot Learning with Discriminative Pre-Trained Models](#prompting-electra-few-shot-learning-with-discriminative-pre-trained-models)
  - [Quick Links](#quick-links)
  - [Overview](#overview)
  - [Reproducibility](#reproducibility)
    - [Requirements](#requirements)
    - [Zero-shot Evaluation](#zero-shot-evaluation)
    - [Few-shot Training](#few-shot-training)
  - [New Datasets and Templates](#new-datasets-and-templates)
  - [Bugs or Questions?](#bugs-or-questions)


## Overview

Pre-trained masked language models have been successfully used for few-shot learning by formulating downstream tasks as text infilling. However, discriminative pre-trained models like ELECTRA, as a strong alternative in full-shot settings, does not fit into the paradigm. In this work, we adapt prompt-based few-shot learning to ELECTRA and show that it outperforms masked language models in a wide range of tasks. ELECTRA is pre-trained to distinguish if a token is generated or original. We naturally extend that to prompt-based few-shot learning by training to score the originality of the verbalizers without introducing new parameters. Our method can be easily adapted to tasks involving multi-token verbalizers without extra computation overhead. Analysis shows that the distributions learned by ELECTRA align better with downstream tasks.


## Reproducibility 

### Requirements
Try runing the following script to install the dependencies.

```bash
pip install -r requirements.txt
```

### Zero-shot Evaluation
We provide scripts for doing zero-shot prompt-based evaluation on the masked language models and BERT and RoBERTa, and discriminative models like ELECTRA. You can find the running script in `scripts/zero_shot_evaluate.sh`. Note that zero-shot evaluation is only apply to tasks with single-token target words.

The arguments are specified as follows:
- `type`: the type of dataset. If it is a glue dataset, type should be specified as `glue`. If the dataset is offered in the datasets package, type should be specified as `other`. If you would like to specify train_file and valid_file, please set it to be `file`.
- `dataset`: the name of the dataset, e.g., sst2, snli.
- `model_name_or_path`: model name or path to the pretrained model, e.g., google/electra-base-discriminator.
- `objective`: please use `dis` for discriminative models, and `mlm` for masked language models.

Also you would need to change the environment variables `main_dir` and `output_dir` to point to the correct directories for logging. An example for doing zero-shot evaluation with an ELECTRA base model on sst2 dataset is as follows:
```bash
bash scripts/zero_shot_evaluate.sh glue sst2 google/electra-base-discriminator dis
```

The expected output is as follows:
```
***** zero shot eval metrics *****
eval_accuracy  =  0.828
```

The zero-shot evaluation results are as follows:

|    | sst2 | sst5 | mr | mnli | snli | rte | qnli | ag_news | boolq
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
BERT | 61.6 | 26.0 | 55.8 | 43.5 | 38.7 | 48.7 | 49.5 | 60.6 | 47.7
RoBERTa | 77.8 | 30.3 | 77.7 | 48.1 | 48.8 | 50.5 | 53.4 | 73.2 | 55.8
ELECTRA | 82.8 | 31.1 | 81.5 | 51.9 | 56.6 | 54.5 | 57.8 | 72.2 | 59.1


### Few-shot Training
For standard few-shot fine-tuning, we provide the script `scripts/few_shot_standard_FT.sh`. The arguments are specified as follows:

- `type`, `dataset`: see the Zero-shot Evaluation section.
- `m`: a shorted name for `model_name_or_path`, please see the script for details.
- `batch_size`: batch_size.
- `lr`: learning rate.
- `k_shot`: number of examples per class in the training set.
- `seed`: random seed to determine the samples for training and validation.

Similarly, you would have to change the environment variables `main_dir` and `output_dir` to point to the correct directories for logging. An example for doing few-shot training with an ELECTRA base model on sst2 dataset is as follows:
```bash
bash scripts/few_shot_standard_FT.sh [TYPE] [DATASET] [M] [BATCH_SIZE] [LR] [K_SHOT] [SEED]
``` 

For prompt-based few-shot fine-tuning, we provide the script `scripts/few_shot_prompt_FT.sh`. Additional arguments include
- ``template_id``: the id of the template, see the object `task_to_template` in `main.py`.

An example is as follows:
```bash
bash scripts/few_shot_prompt_FT.sh [TYPE] [DATASET] [M] [BATCH_SIZE] [LR] [K_SHOT] [SEED] [TEMPLATE_ID]
``` 

For few-shot experiments involving multi-token options, we provide the script `scripts/few_shot_span_prompt_FT.sh`. Additional arguments include

- `span_rep_type`: the type of span representation. `cls` denotes using the representation of the CLS head. `prob` denotes using the average probability of an option and `average` denotes using the average representations of all tokens of an span for discrimination.
- `discriminator_head`: `pretrained` denotes using the pretrained discriminator head, `new` denotes training the discriminator head from scratch.

For each experiment with the few-shot setting, we run with three different random seeds and for each random seed, we do hyperparameter search on batch sizes `2, 4, 8` and learning rates `1e-5, 2e-5, 3e-5`. 


## New Datasets and Templates
All the datasets except `mr` and `sst5` can be loaded through the datasets package. For `mr` and `sst5`, please download the datasets through [this link](https://github.com/AcademiaSinicaNLPLab/sentiment_dataset). To add a new dataset for evaluation, make sure your either load the dataset through `datasets.load_dataset(name)` or specify `train_file` and `validation_file` in the scripts. You also need to add a corresponding template in `templates.py` and link the task to the template in `main.py`.

## License
The majority of ELECTRA-Fewshot-Learning is licensed under CC-BY-NC, however portions of the project are available under separate license terms: https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification is licensed under the Apache 2.0 license.

## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Mengzhou (mengzhou@princeton.edu). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

<!-- ## Citation

Please cite our paper with the following bibtex:

```bibtex
   title={Prompting ELECTRA: Few-Shot Learning with Discriminative
Pre-Trained Models},
   author={Xia, Mengzhou and Artetxe, Mikel and Du, Jingfei and Chen, Danqi and Stoyanov, Veselin},
   booktitle={Association for Computational Linguistics (ACL)},
   year={2022}
}
``` -->