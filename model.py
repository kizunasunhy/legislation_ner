from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertModel, RobertaPreTrainedModel, RobertaModel
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.nn as nn

class Bert_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

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
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            #crf_map = (labels != -1).float()
            #crf_map = crf_map.type(torch.uint8)
            #print(input_ids[0], labels[0], crf_map[0])
            
            # Only keep active parts of the loss
            if attention_mask is not None:
                attention_mask = attention_mask.type(torch.uint8)
                log_likelihood = self.crf(logits, labels, attention_mask)
                loss = -1 * log_likelihood
            else:    
                log_likelihood = self.crf(logits, labels)
                loss = -1 * log_likelihood
    
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    
class Roberta_CRF(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

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
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            #crf_map = (labels != -1).float()
            #crf_map = crf_map.type(torch.uint8)
            #print(input_ids[0], labels[0], crf_map[0])
            
            # Only keep active parts of the loss
            if attention_mask is not None:
                attention_mask = attention_mask.type(torch.uint8)
                log_likelihood = self.crf(logits, labels, attention_mask)
                loss = -1 * log_likelihood
            else:    
                log_likelihood = self.crf(logits, labels)
                loss = -1 * log_likelihood
    
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    