# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class AdditionalArguments():
    FT_method: str = field(default="sequence_classification", metadata={"help": "prompt_tuning, sequence_classification"})
    objective: str = field(default="mlm", metadata={"help": "mlm, "})
    k_shot: int = field(default=0, metadata={"help": "k-shot learning."})
    add_task_name: str = field(default="sst5", metadata={"help": "For tasks passing with train file and dev file"})
    do_zero_shot_eval: bool = field(default=False)

    span_rep_type: str = field(default="average", metadata={"help": ['average', "start_end"]})
    from_nonsym_to_sym_dis: bool = field(default=False)

    do_analysis: bool = field(default=False)
    analysis_version: int = field(default=1)
    data_seed: int = field(default=42)

    template_id: int = field(default=2)
    discriminator_head: str = field(default="pretrained", metadata={"help": "pretrained, new"})



