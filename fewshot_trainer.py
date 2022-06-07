# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.
import pdb
from pyrsistent import freeze

from transformers.trainer import Trainer
import collections
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers.trainer import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    PushToHubMixin,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
from transformers.optimization import Adafactor, AdamW, get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    number_of_arguments,
    set_seed,
    speed_metrics,
)
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import logging
from torch.utils.data import Sampler

logger = logging.get_logger(__name__)
_is_torch_generator_available = False
if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True

class DisRandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None, num_labels=2) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self.num_labels=2

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source) // self.num_labels
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        l = torch.randperm(n, generator=generator)
        l = torch.stack([l+i for i in range(self.num_labels)])
        l = l.transpose(0, 1).reshape(-1).tolist()
        yield from l

    def __len__(self) -> int:
        return self.num_samples

class MetricTracker():
    def __init__(self, metric_name="accuracy"):
        self.epoch = 0
        self.global_step = 0
        self.best_eval_score = 0
        self.metric = metric_name
        self.all_scores = None

    def format_score(self, score):
        if str(score).startswith("0."):
            score = round(score * 100, 2)
        else:
            score = round(score, 2)
        return score

    def update(self, epoch, global_step, eval_score, all_scores):
        eval_score = self.format_score(eval_score)
        if eval_score > self.best_eval_score:
            self.epoch = epoch
            self.global_step = global_step
            self.best_eval_score = eval_score
            self.all_scores = all_scores
            print(f"Epoch {self.epoch} Step {global_step}: Best score ({self.metric}) so far: {eval_score}")

    def clear(self):
        self.eval_score = 0


class FewShotTrainer(Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            mask_idx: int = None,
            static_input: Dict = None,
            additional_args=None,
            task_metric="accuracy",
            objective=None,
    ):
        super(FewShotTrainer, self).__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers)
        self.static_input = static_input
        self.mask_idx = mask_idx
        self.additional_args = additional_args
        self.task_metric = task_metric
        self.metric_tracker = MetricTracker(task_metric)
        self.objective = objective # determine the sampler

    # for freezing the encoder
    def create_optimizer(self):
        if self.optimizer is None:
            if self.additional_args.FT_method == "linear_prob":
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if "prob_classifier" in n],
                        "weight_decay": self.args.weight_decay,
                    },
                ] 
            elif self.additional_args.FT_method == "linear_fit":
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if "linear_fit" in n],
                        "weight_decay": self.args.weight_decay,
                    },
                ] 
            else:
                decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
                decay_parameters = [name for name in decay_parameters if "bias" not in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if not isinstance(self.train_dataset, collections.abc.Sized):
            return None

        generator = None
        if self.args.world_size <= 1 and _is_torch_generator_available:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        sampler = DisRandomSampler if self.objective == "parallel_dis" else RandomSampler 
        if _is_torch_generator_available:
            return sampler(self.train_dataset, generator=generator)
        return sampler(self.train_dataset)
        

    def zero_shot_evaluate(self):
        # if self.additional_args == ",l,"
        #     return self.evaluate(eval_dataset)
        # else:
        return self.evaluate()

    def shorten_input(self, inputs):
        max_length = inputs["attention_mask"].sum(-1).max().item()
        inputs["input_ids"] = inputs["input_ids"][:, :max_length]
        inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]
        if "token_type_ids" in inputs:
            inputs["token_type_ids"] = inputs["token_type_ids"][:, :max_length]

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        self.shorten_input(inputs)
        if self.static_input is not None:
            inputs.update(self.static_input)

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self._memory_tracker.start()

        metrics = None

        for key in self.eval_dataset:
            eval_d = self.eval_dataset[key]
            eval_dataloader = self.get_eval_dataloader(eval_d)
            start_time = time.time()

            eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

            total_batch_size = self.args.eval_batch_size * self.args.world_size
            output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            ))
            updating_metrics = {f"{key}_{k}": output.metrics[k] for k in output.metrics}
            if metrics is None:
                metrics = updating_metrics
            else:
                metrics.update(updating_metrics) 
                       

        self.log(metrics)
        eval_key = "validation"
        self.metric_tracker.update(self.state.epoch, self.state.global_step, metrics[f"{eval_key}_{metric_key_prefix}_{self.task_metric}"], metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics


    def analyze_v1(
        self,
        description: str, 
    ):
        all_additional_info = {}
        for eval_dataset_name in ["fewshot_validation"]:
            dataloader = self.get_eval_dataloader(self.eval_dataset[eval_dataset_name])
            model = self._wrap_model(self.model, training=False)

            batch_size = dataloader.batch_size

            logger.info(f"***** Running {description} *****")
            if isinstance(dataloader.dataset, collections.abc.Sized):
                logger.info(f"  Num examples = {self.num_examples(dataloader)}")
            else:
                logger.info("  Num examples: Unknown")
            logger.info(f"  Batch size = {batch_size}")

            model.eval()

            self.callback_handler.eval_dataloader = dataloader
            from collections import defaultdict
            additional_info = defaultdict(list)
        
            # Main evaluation loop
            for step, inputs in enumerate(dataloader):
                # Prediction step
                inputs = self._prepare_inputs(inputs)
                additional_info_step = self.analysis_v1_step(model, inputs)
                for key in ["logits", "labels"]: # additional_info_step:
                    additional_info[key].append(additional_info_step[key].cpu().numpy())
                if (step+1) % 100 == 0:
                    logger.info(f"Processed {step+1} batches")
                import pdb; pdb.set_trace()
            for key in additional_info:
                additional_info[key] = np.concatenate(additional_info[key], axis=0)
            all_additional_info[eval_dataset_name] = additional_info
        torch.save(all_additional_info, os.path.join(self.args.output_dir, f"additional-info.pt"))

    # getting electra's reps and reps after discriminator head
    def analyze_v2(
        self,
        description: str, 
    ):
        reps = {}
        for eval_dataset_name in self.eval_dataset:
            dataloader = self.get_eval_dataloader(self.eval_dataset[eval_dataset_name])
            model = self._wrap_model(self.model, training=False)

            batch_size = dataloader.batch_size

            logger.info(f"***** Running {description} *****")
            if isinstance(dataloader.dataset, collections.abc.Sized):
                logger.info(f"  Num examples = {self.num_examples(dataloader)}")
            else:
                logger.info("  Num examples: Unknown")
            logger.info(f"  Batch size = {batch_size}")

            model.eval()

            self.callback_handler.eval_dataloader = dataloader
            from collections import defaultdict
            rep = defaultdict(list)
        
            # Main evaluation loop
            for step, inputs in enumerate(dataloader):
                # Prediction step
                print(step)
                inputs = self._prepare_inputs(inputs)
                with torch.no_grad():
                    self.shorten_input(inputs)
                    inputs["return_reps"] = True
                    rep_step = model(**inputs)
                for key in rep_step:
                    rep_step_key = rep_step[key].cpu().numpy()
                    rep_step_key = rep_step_key.reshape(-1, rep_step_key.shape[-1])
                    rep[key].append(rep_step_key)
            for key in rep:
                rep[key] = np.concatenate(rep[key], axis=0)
            reps[eval_dataset_name] = rep
        torch.save(reps, os.path.join(self.args.output_dir, f"reps.pt"))

    def analysis_v1_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        inputs["return_keywords_reps"] = True
        with torch.no_grad():
            self.shorten_input(inputs)
            if self.static_input is not None:
                inputs.update(self.static_input)
            return model(**inputs)

    # for linear fitting
    # def prediction_step(
    #     self,
    #     model: nn.Module,
    #     inputs: Dict[str, Union[torch.Tensor, Any]],
    #     prediction_loss_only: bool,
    #     ignore_keys: Optional[List[str]] = None,
    # ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    #     inputs = self._prepare_inputs(inputs)

    #     with torch.no_grad():
    #         loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
    #         loss = loss.mean().detach()
    #         logits, labels = outputs[1:] 
    #     return (loss, logits, labels) 