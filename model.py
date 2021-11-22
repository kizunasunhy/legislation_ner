from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertModel, RobertaPreTrainedModel, RobertaModel
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.nn as nn
from linears import PoolerEndLogits, PoolerStartLogits
from focal_loss import FocalLoss
from label_smoothing import LabelSmoothingCrossEntropy

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

class RobertaForTokenClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier  = nn.Linear(config.hidden_size, config.num_labels)

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
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
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

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=-100)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=-100)
            elif self.loss_type == 'ce':
                loss_fct = CrossEntropyLoss(ignore_index=-100)  
            #loss_fct = CrossEntropyLoss()
            
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    
class BertSpanForNer(BertPreTrainedModel):
    def __init__(self, config,):
        super(BertSpanForNer, self).__init__(config)
        self.soft_label = config.soft_label
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None,end_positions=None,labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        
        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        outputs = (start_logits, end_logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type =='lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)
            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_positions.view(-1)[active_loss]
            active_end_labels = end_positions.view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs
    
class RobertaSpanNer(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.dense1 = nn.Linear(config.hidden_size, config.num_labels)
        self.dense2 = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
          
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
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

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        start_logits = self.dense1(sequence_output)
        end_logits = self.dense2(sequence_output)

        total_loss = None
        if labels is not None:
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=-100)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=-100)
            elif self.loss_type == 'ce':
                loss_fct = CrossEntropyLoss(ignore_index=-100)  
            #loss_fct = CrossEntropyLoss()
            
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits1 = start_logits.view(-1, self.num_labels)
                active_logits2 = end_logits.view(-1, self.num_labels)
                start_active_labels = torch.where(
                    active_loss, start_positions.view(-1), torch.tensor(loss_fct.ignore_index).type_as(start_positions)
                )
                end_active_labels = torch.where(
                    active_loss, end_positions.view(-1), torch.tensor(loss_fct.ignore_index).type_as(end_positions)
                )
                start_loss = loss_fct(active_logits1, start_active_labels)
                end_loss = loss_fct(active_logits2, end_active_labels)
            else:
                start_loss = loss_fct(start_logits.view(-1, self.num_labels), start_positions.view(-1))
                end_loss = loss_fct(end_logits.view(-1, self.num_labels), end_positions.view(-1))
            total_loss = (start_loss + end_loss)/2
            
        output = (start_logits, end_logits,) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output