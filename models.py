# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
import math
import pdb
from turtle import position
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import ElectraForSequenceClassification, ElectraForMaskedLM

from transformers.activations import ACT2FN, gelu
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    ModelOutput
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM, RobertaForSequenceClassification, RobertaPreTrainedModel, RobertaModel
from transformers.models.bert.modeling_bert import BertForMaskedLM, BertForSequenceClassification, BertModel, BertPreTrainedModel
from transformers.models.electra.modeling_electra import ElectraForPreTraining, ElectraForPreTrainingOutput, ElectraPreTrainedModel, ElectraModel, ElectraClassificationHead
from transformers.models.bert.modeling_bert import BertForMaskedLM, BertLMPredictionHead
from transformers.activations import ACT2FN, get_activation

class ElectraDiscriminatorSpanPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        if config.span_rep_type == "start_end":
            self.replaced_dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, discriminator_hidden_states):
        if hasattr(self, "dense"):
            hidden_states = self.dense(discriminator_hidden_states)
        else:
            hidden_states = self.replaced_dense(discriminator_hidden_states) 
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)

        return logits

class ElectraForPromptSpanTraining(ElectraForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        if config.discriminator_head == "new":
            self.discriminator_predictions_scratch = ElectraDiscriminatorSpanPredictions(config)
        else:
            self.discriminator_predictions = ElectraDiscriminatorSpanPredictions(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        token_start=None,
        token_end=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        discriminator_hidden_states = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]
        token_start = token_start.squeeze()
        token_end = token_end.squeeze()
        bs = len(input_ids)
        if self.config.span_rep_type == "start_end":
            start_reps = discriminator_sequence_output[range(bs), token_start]
            end_reps = discriminator_sequence_output[range(bs), token_end]
            reps = torch.cat([start_reps, end_reps], dim=-1)
        elif self.config.span_rep_type == "average":
            reps = []
            for i in range(bs):
                st = token_start[i]
                en = token_end[i]
                rep = discriminator_sequence_output[i][st:en].mean(dim=0)
                reps.append(rep)
            reps = torch.stack(reps)
        elif self.config.span_rep_type == "cls": # take the CLS token
            reps = discriminator_sequence_output[:, 0] 
        else:
            reps = discriminator_sequence_output

        if self.config.discriminator_head == "new":
            logits = self.discriminator_predictions_scratch(reps)
        else:
            logits = self.discriminator_predictions(reps)
        # -0.1293,  0.1417,  0.1912,  0.9741, -0.0792,  0.1598,  0.1294,  0.1712]]
        selected_logits = []
        if self.config.span_rep_type == "max_prob":
            for i in range(bs):
                st = token_start[i]
                en = token_end[i]
                prob = torch.max(logits[i][st:en])
                selected_logits.append(prob)
            logits = torch.stack(selected_logits)
        elif self.config.span_rep_type == "mean_prob":
            for i in range(bs):
                st = token_start[i]
                en = token_end[i]
                prob = torch.mean(logits[i][st:en])
                selected_logits.append(prob)
            logits = torch.stack(selected_logits)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                # active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                # active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                # active_labels = labels[active_loss]
                # loss = loss_fct(active_logits, active_labels.float())
                loss = loss_fct(logits, labels.float())
            else:
                loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), labels.float())

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return ElectraForPreTrainingOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )




class ElectraForPromptSoftmaxTuning(ElectraForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.ffn_head = FFNHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        token_start=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]
        token_start = token_start.repeat(discriminator_sequence_output.shape[-1], 1, 1).transpose(0, 1).transpose(1, 2)
        discriminator_sequence_output = torch.gather(discriminator_sequence_output, 1, token_start)
        prediction_scores = self.ffn_head(discriminator_sequence_output).squeeze(-1)  # batch_size * num_labels

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(prediction_scores, labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + discriminator_hidden_states[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

# Electra with the contrastive loss
class NewElectraDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, discriminator_hidden_states, return_keywords_reps=False):
        additional_info = {}
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)
        if return_keywords_reps:
            additional_info["hidden_states"] = hidden_states
            additional_info["logits"] = logits
            return additional_info
        return logits

class ElectraV2ForPromptTuning(ElectraForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.discriminator_predictions = NewElectraDiscriminatorPredictions(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        token_start=None,
        return_keywords_reps=False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]
        token_start = token_start.squeeze()

        # if return_keywords_reps:
            # return discriminator_sequence_output[range(len(input_ids)), token_start]

        logits = self.discriminator_predictions(discriminator_sequence_output, return_keywords_reps)
        
        if return_keywords_reps:
            return logits[range(len(input_ids)), token_start] 
        
        logits = logits[range(len(input_ids)), token_start].reshape(-1) # batch_size
        logits = -logits.reshape(-1, len(self.config.id2label))
        labels = torch.argmin(labels.reshape(-1, len(self.config.id2label)), dim=-1)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                # active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                # active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                # active_labels = labels[active_loss]
                # loss = loss_fct(active_logits, active_labels.float())
                loss = loss_fct(logits, labels.view(-1))
            else:
                loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), labels.float())

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return ElectraForPreTrainingOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


class ElectraForPromptTuning(ElectraForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.discriminator_predictions = NewElectraDiscriminatorPredictions(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        token_start=None,
        return_keywords_reps=False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]
        token_start = token_start.squeeze()

        # if return_keywords_reps:
            # return discriminator_sequence_output[range(len(input_ids)), token_start]

        logits = self.discriminator_predictions(discriminator_sequence_output, return_keywords_reps)
        
        if return_keywords_reps:
            logits["hidden_states"] = logits["hidden_states"][range(len(input_ids)), token_start] 
            logits["logits"] = logits["logits"][range(len(input_ids)), token_start].reshape(-1) 
            logits["labels"] = labels
            additional_info = logits
            return additional_info 
        
        logits = logits[range(len(input_ids)), token_start].reshape(-1) # batch_size

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                # active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                # active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                # active_labels = labels[active_loss]
                # loss = loss_fct(active_logits, active_labels.float())
                loss = loss_fct(logits, labels.float())
            else:
                loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), labels.float())

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return ElectraForPreTrainingOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

@dataclass
class NewElectraForPreTrainingOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    keyword_reps: Optional[torch.FloatTensor] = None


class ElectraForPromptSoftmaxTuning(ElectraForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.ffn_head = FFNHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        token_start=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]
        token_start = token_start.repeat(discriminator_sequence_output.shape[-1], 1, 1).transpose(0, 1).transpose(1, 2)
        discriminator_sequence_output = torch.gather(discriminator_sequence_output, 1, token_start)
        prediction_scores = self.ffn_head(discriminator_sequence_output).squeeze(-1)  # batch_size * num_labels

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(prediction_scores, labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + discriminator_hidden_states[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

class NewRobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, return_keywords_reps=False):
        x = self.dense(features)
        x = gelu(x)
        hidden_states = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(hidden_states)
        additional_info = {}
        if return_keywords_reps:
            additional_info["hidden_states"] = hidden_states
            additional_info["logits"] = x
            return additional_info
        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class RobertaForPromptSpanTraining(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.discriminator_predictions = ElectraDiscriminatorSpanPredictions(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            token_start=None,
            token_end=None
    ):
       
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        token_start = token_start.squeeze()
        token_end = token_end.squeeze()
        bs = len(input_ids)
        if self.config.span_rep_type == "start_end":
            start_reps = sequence_output[range(bs), token_start]
            end_reps = sequence_output[range(bs), token_end]
            reps = torch.cat([start_reps, end_reps], dim=-1)
        elif self.config.span_rep_type == "average":
            reps = []
            for i in range(bs):
                st = token_start[i]
                en = token_end[i]
                rep = sequence_output[i][st:en].mean(dim=0)
                reps.append(rep)
            reps = torch.stack(reps)
        elif self.config.span_rep_type == "cls":
            reps = sequence_output[:, 0]

        logits = self.discriminator_predictions(reps)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                # active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                # active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                # active_labels = labels[active_loss]
                # loss = loss_fct(active_logits, active_labels.float())
                loss = loss_fct(logits, labels.float())
            else:
                loss = loss_fct(logits.view(-1, sequence_output.shape[1]), labels.float())

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ElectraForPreTrainingOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class ElectraGeneratorForPromptTuning(ElectraForMaskedLM):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        inspect_indexes = None,
        mask_idx = None,
        return_keywords_reps=False,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        generator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        generator_sequence_output = generator_hidden_states[0]
        mask_pos = (input_ids == mask_idx)
        generator_sequence_output = generator_sequence_output[mask_pos]

        prediction_scores = self.generator_predictions(generator_sequence_output)
        prediction_scores = self.generator_lm_head(prediction_scores)
        prediction_scores = prediction_scores[:, inspect_indexes]
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, len(inspect_indexes)), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + generator_hidden_states[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=generator_hidden_states.hidden_states,
            attentions=generator_hidden_states.attentions,
        )


class RobertaForPromptTuning(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.lm_head = NewRobertaLMHead(config)


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            mask_idx=None,
            inspect_indexes=None,
            return_keywords_reps=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        mask_pos = (input_ids == mask_idx)
        sequence_output = sequence_output[mask_pos]
        prediction_scores = self.lm_head(sequence_output, return_keywords_reps)
        if return_keywords_reps:
            prediction_scores["hidden_states"] = prediction_scores["hidden_states"]
            prediction_scores["logits"] = prediction_scores["logits"][:, inspect_indexes].reshape(-1) 
            prediction_scores["labels"] = labels
            additional_info = prediction_scores
            return additional_info 
        prediction_scores = prediction_scores[:, inspect_indexes]
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, len(inspect_indexes)), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FFNHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, 1)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


class RobertaForPromptSoftmaxTuning(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.ffn_head = FFNHead(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            token_start=None,
            token_end=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        # only supports single label words
        token_start = token_start.repeat(sequence_output.shape[-1], 1, 1).transpose(0, 1).transpose(1, 2)
        sequence_output = torch.gather(sequence_output, 1, token_start)
        prediction_scores = self.ffn_head(sequence_output).squeeze(-1) # batch_size * num_labels

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(prediction_scores, labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class NewBertLMPredictionHead(BertLMPredictionHead):
    def forward(self, hidden_states, return_keywords_reps=False):
        hidden_states = self.transform(hidden_states)
        logits = self.decoder(hidden_states)
        if return_keywords_reps:
            additional_info = {"hidden_states": hidden_states}
            additional_info["logits"] = logits
            return additional_info 
        return logits 

class NewBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = NewBertLMPredictionHead(config)

    def forward(self, sequence_output, return_keywords_reps=False):
        prediction_scores = self.predictions(sequence_output, return_keywords_reps)
        return prediction_scores

class BertForPromptTuning(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.cls = NewBertOnlyMLMHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mask_idx=None,
        inspect_indexes=None,
        return_keywords_reps=None,
        token_start=None,
        
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        
        # if return_keywords_reps:
            # return sequence_output[range(len(input_ids)), token_start]

        mask_pos = (input_ids == mask_idx)
        sequence_output = sequence_output[mask_pos]
        # if return_keywords_reps:
            # return sequence_output
        
        prediction_scores = self.cls(sequence_output, return_keywords_reps)
        if return_keywords_reps:
            prediction_scores["hidden_states"] = prediction_scores["hidden_states"]
            prediction_scores["logits"] = prediction_scores["logits"][:, inspect_indexes].reshape(-1) 
            prediction_scores["labels"] = labels
            additional_info = prediction_scores
            return additional_info 
        prediction_scores = prediction_scores[:, inspect_indexes]

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, len(inspect_indexes)), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertForPromptSoftmaxTuning(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.ffn_head = FFNHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        token_start=None,
        token_end=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # only supports single label words
        token_start = token_start.repeat(sequence_output.shape[-1], 1, 1).transpose(0, 1).transpose(1, 2)
        sequence_output = torch.gather(sequence_output, 1, token_start)
        prediction_scores = self.ffn_head(sequence_output).squeeze(-1)  # batch_size * num_labels

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(prediction_scores, labels.view(-1))


        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class ElectraForLinearProb(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.electra = ElectraModel(config)
        self.num_labels = config.num_labels
        self.config = config
        self.prob_classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_reps=False,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = discriminator_hidden_states[0]
        sequence_output = sequence_output[:, 0]
        logits = self.prob_classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )



class BertForLinearProb(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.prob_classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        last_hidden_states = outputs[0][:, 0]

        logits = self.prob_classifier(last_hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class ElectraClassificationHeadStandardAnalysis(ElectraClassificationHead):
    """Head for sentence-level classification tasks."""

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        # if kwargs["return_reps"]:
        # x = self.dropout(x)
        # x = self.out_proj(x)
        return x

class ElectraForSequenceClassificationStandardAnalysis(ElectraForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = ElectraClassificationHeadStandardAnalysis(config)
        self.linear_fit = nn.Linear(config.hidden_size, config.hidden_size)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_reps=False,
    ) -> Union[Tuple, SequenceClassifierOutput]:
       
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = discriminator_hidden_states[0]
        head_hidden_layer = self.classifier(sequence_output, return_reps=return_reps)
        predicted_head_hidden_layer = self.linear_fit(sequence_output)
        # batch * seq * 768
        predicted_head_hidden_layer = predicted_head_hidden_layer * attention_mask.unsqueeze(-1)
        head_hidden_layer = head_hidden_layer * attention_mask.unsqueeze(-1)
        loss_fct = MSELoss(reduction="none")
        
        loss = loss_fct(predicted_head_hidden_layer, head_hidden_layer)
        numerator = loss.sum(-1).sum(-1) 
        denominator = head_hidden_layer.pow(2).sum(-1).sum(-1)

        loss = loss.sum() / torch.sum(attention_mask)

        return RegressionOutput(
            loss=loss,
            numerator=numerator,
            denominator=denominator
        )

@dataclass
class RegressionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    numerator: torch.FloatTensor = None
    denominator: torch.FloatTensor = None