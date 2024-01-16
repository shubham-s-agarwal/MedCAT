import torch
from collections import OrderedDict
from typing import Optional, Any, List
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers.modeling_outputs import TokenClassifierOutput
from medcat.meta_cat import ConfigMetaCAT

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

class LSTM(nn.Module):
    def __init__(self, embeddings: Optional[Tensor], config: ConfigMetaCAT) -> None:
        super(LSTM, self).__init__()

        self.config = config
        # Get the required sizes
        vocab_size = config.general['vocab_size']
        embedding_size = config.model['input_size']

        # Initialize embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=config.model['padding_idx'])
        if embeddings is not None:
            self.embeddings.load_state_dict(OrderedDict([('weight', embeddings)]))
            # Disable training for the embeddings - IMPORTANT
            self.embeddings.weight.requires_grad = config.model['emb_grad']

        # Create the RNN cell - devide
        self.rnn = nn.LSTM(input_size=config.model['input_size'],
                           hidden_size=config.model['hidden_size'] // config.model['num_directions'],
                           num_layers=config.model['num_layers'],
                           dropout=config.model['dropout'],
                           bidirectional=config.model['num_directions'] == 2)
        self.fc1 = nn.Linear(config.model['hidden_size'], config.model['nclasses'])

        self.d1 = nn.Dropout(config.model['dropout'])

    def forward(self,
                input_ids: torch.LongTensor,
                center_positions: Tensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                ignore_cpos: bool = False,model_arch_config=None) -> Tensor:
        x = input_ids
        # Get the mask from x
        if attention_mask is None:
            mask = x != self.config.model['padding_idx']
        else:
            mask = attention_mask

        # Embed the input: from id -> vec
        x = self.embeddings(x) # x.shape = batch_size x sequence_length x emb_size

        # Tell RNN to ignore padding and set the batch_first to True
        x = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).int().view(-1).cpu(), batch_first=True, enforce_sorted=False)

        # Run 'x' through the RNN
        x, hidden = self.rnn(x)

        # Add the padding again
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Get what we need
        row_indices = torch.arange(0, x.size(0)).long()

        # If this is  True we will always take the last state and not CPOS
        if ignore_cpos:
            x = hidden[0]
            x = x.view(self.config.model['num_layers'], self.config.model['num_directions'], -1,
                       self.config.model['hidden_size']//self.config.model['num_directions'])
            x = x[-1, :, :, :].permute(1, 2, 0).reshape(-1, self.config.model['hidden_size'])
        else:
            x = x[row_indices, center_positions, :]

        # Push x through the fc network and add dropout
        x = self.d1(x)
        x = self.fc1(x)

        return x

# class BertForMetaAnnotation(BertPreTrainedModel):
#
#     _keys_to_ignore_on_load_unexpected: List[str] = [r"pooler"]  # type: ignore
#
#     def __init__(self, config: BertConfig) -> None:
#         bert_config = BertConfig.from_dict(dict(config.model))
#         super().__init__(bert_config)
#         self.num_labels = config.model["nclasses"]
#
#         self.bert = BertModel(bert_config, add_pooling_layer=False)
#         self.dropout = nn.Dropout(config.model.dropout)
#         self.classifier = nn.Linear(config.model.hidden_size, config.model.nclasses)
#
#         self.init_weights() # type: ignore
#
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         center_positions: Optional[Any] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> TokenClassifierOutput:
#         """labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
#             Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
#             1]``.
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict # type: ignore
#
#         outputs = self.bert( # type: ignore
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             #return_dict=return_dict,
#         )
#
#         sequence_output = outputs[0] # (batch_size, sequence_length, hidden_size)
#         print(sequence_output.shape)
#
#         row_indices = torch.arange(0, sequence_output.size(0)).long()
#         sequence_output = sequence_output[row_indices, center_positions, :]
#
#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)
#
#         loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             # Only keep active parts of the loss
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#
#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output
#
#         return TokenClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


class BertForMetaAnnotation(nn.Module):

    _keys_to_ignore_on_load_unexpected: List[str] = [r"pooler"]  # type: ignore

    def __init__(self,config,model_arch_config):
        super(BertForMetaAnnotation, self).__init__()

        bert = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels= config.model["nclasses"]
        )

        self.bert = bert
        self.num_labels = config.model["nclasses"]
        print("Nclasses:",self.num_labels)
        print("Architecture config received",model_arch_config)
        for param in self.bert.parameters():
            param.requires_grad = not(config.model.model_freeze_layers)

        hidden_size_2 = int(config.model.hidden_size/2)
        # dropout layer
        self.dropout = nn.Dropout(config.model.dropout)
        # relu activation function
        self.relu = nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(config.model['input_size'], config.model.hidden_size)
        # dense layer 2
        self.fc2 = nn.Linear(config.model.hidden_size, hidden_size_2)
        # dense layer 3
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_2)
        # dense layer 3 (Output layer)
        if model_arch_config is not None:
            if model_arch_config['fc2'] is True or model_arch_config['fc3'] is True:
                self.fc4 = nn.Linear(hidden_size_2, self.num_labels)
            else:
                self.fc4 = nn.Linear(config.model.hidden_size, self.num_labels)
        else:
            self.fc4 = nn.Linear(hidden_size_2, self.num_labels)
        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        center_positions: Optional[Any] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        model_arch_config=None
    ):
        """labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict # type: ignore

        outputs = self.bert( # type: ignore
            input_ids,
            attention_mask=attention_mask,output_hidden_states=True
        )

        row_indices = torch.arange(0, outputs.hidden_states[0].size(0)).long()
        x = outputs.hidden_states[0][row_indices, center_positions, :]
        #x = outputs.hidden_states[0][:, 0, :]  # To retrieve the [CLS] token that holds all learned context from BERT
        #fc1
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        if model_arch_config is not None:
            if model_arch_config['fc2'] is True:
                # fc2
                x = self.fc2(x)
                x = self.relu(x)
                x = self.dropout(x)

            if model_arch_config['fc3'] is True:
                # fc3
                x = self.fc3(x)
                x = self.relu(x)
                x = self.dropout(x)
        else:
            # fc2
            x = self.fc2(x)
            x = self.relu(x)
            x = self.dropout(x)

            #fc3
            x = self.fc3(x)
            x = self.relu(x)
            x = self.dropout(x)

        # output layer
        x = self.fc4(x)
        # apply softmax activation
        x = self.softmax(x)
        return x