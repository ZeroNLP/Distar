from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, AnyStr, Tuple, List, Union

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import (BertPreTrainedModel,
                          RobertaPreTrainedModel,
                          BertConfig,
                          RobertaConfig,
                          BertModel,
                          RobertaModel,
                          BertTokenizerFast,
                          RobertaTokenizerFast)
from transformers.modeling_outputs import ModelOutput, BaseModelOutputWithPoolingAndCrossAttentions

from .kge_scorer import TransE, DistMult, ComplEx, RotatE


@dataclass
class DistarOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    logits: torch.FloatTensor = None
    kge_loss: torch.FloatTensor = None
    contra_loss: torch.FloatTensor = None
    kge_probs: torch.FloatTensor = None


class BertCrfKge(BertPreTrainedModel):
    def __init__(self,
                 config: Optional[BertConfig] = None,
                 need_lstm: Optional[bool] = True,
                 lstm_dim: Optional[int] = 128,
                 num_lstm_layer: Optional[int] = 1,
                 num_role_encoder_layers: Optional[int] = 2,
                 kge_scorer_name: Optional[AnyStr] = "TransE",
                 triplet_comb: Optional[AnyStr] = "ar_t"
                 ) -> None:
        super(BertCrfKge, self).__init__(config)

        self.need_lstm = need_lstm
        self.lstm_dim = lstm_dim
        self.num_lstm_layer = num_lstm_layer
        self.num_role_encoder_layers = num_role_encoder_layers
        self.kge_scorer_name = kge_scorer_name
        self.triplet_comb = triplet_comb

        # argument identification module
        self.bert = BertModel(config=config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        out_dim = config.hidden_size

        self.lstm = None
        if self.need_lstm:
            out_dim = self.lstm_dim * 2
            self.lstm = nn.LSTM(input_size=config.hidden_size,
                                hidden_size=self.lstm_dim,
                                num_layers=self.num_lstm_layer,
                                bidirectional=True,
                                batch_first=True)

        self.classifier = nn.Linear(in_features=out_dim, out_features=config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        # argument classification module
        kge_config = deepcopy(config)
        kge_config.num_hidden_layers = self.num_role_encoder_layers
        self.kge_model = BertModel(config=kge_config)
        self.kge_model.embeddings = self.bert.embeddings
        self.sigmoid = nn.Sigmoid()
        self.kge_criterion = nn.BCEWithLogitsLoss()

        self.gamma = config.hidden_size // 2
        self.kge_scorer = None
        if self.kge_scorer_name == "TransE":
            self.kge_scorer = TransE(self.gamma)
        elif self.kge_scorer_name == "DistMult":
            self.kge_scorer = DistMult()
        elif self.kge_scorer_name == "ComplEx":
            self.kge_scorer = ComplEx()
        elif self.kge_scorer_name == "RotatE":
            self.kge_scorer = RotatE(self.gamma, config.hidden_size)
        else:
            raise ValueError(f"no module called {self.kge_scorer_name}")

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                trigger_mask: Optional[torch.Tensor] = None,
                neg_trigger_mask: Optional[torch.Tensor] = None,
                argument_mask: Optional[torch.Tensor] = None,
                role_input_ids: Optional[torch.Tensor] = None,
                role_attention_mask: Optional[torch.Tensor] = None,
                role_token_type_ids: Optional[torch.Tensor] = None,
                triplet_label: Optional[torch.Tensor] = None
                ) -> DistarOutput:
        # argument identification module forward
        logits, bert_outputs = self.identification_module_forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        loss = None
        if labels is not None:
            loss = -1.0 * self.crf(emissions=logits,
                                   tags=labels,
                                   mask=attention_mask.bool())
            loss = loss / input_ids.shape[0]

        # argument classification module forward
        kge_loss = None
        kge_score = None
        if role_input_ids is not None:
            kge_score = self.classification_module_forward(
                trigger_mask=trigger_mask,
                neg_trigger_mask=neg_trigger_mask,
                argument_mask=argument_mask,
                role_input_ids=role_input_ids,
                role_attention_mask=role_attention_mask,
                role_token_type_ids=role_token_type_ids,
                bert_outputs=bert_outputs
            )
            triplet_label = triplet_label.view((-1,))
            kge_loss = self.kge_criterion(kge_score, triplet_label)
            loss += kge_loss

        return DistarOutput(
            loss=loss,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
            logits=logits,
            kge_loss=kge_loss,
            kge_probs=self.sigmoid(kge_score) if kge_score is not None else None,
        )

    def identification_module_forward(self,
                                      input_ids: Optional[torch.Tensor] = None,
                                      token_type_ids: Optional[torch.Tensor] = None,
                                      attention_mask: Optional[torch.Tensor] = None
                                      ) -> Tuple[torch.FloatTensor,
                                                 BaseModelOutputWithPoolingAndCrossAttentions]:
        bert_outputs = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 return_dict=True)
        sequence_output = bert_outputs.last_hidden_state

        if self.need_lstm:
            sequence_output = self.lstm(sequence_output)[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits, bert_outputs

    def classification_module_forward(self,
                                      trigger_mask: Optional[torch.Tensor] = None,
                                      neg_trigger_mask: Optional[torch.Tensor] = None,
                                      argument_mask: Optional[torch.Tensor] = None,
                                      role_input_ids: Optional[torch.Tensor] = None,
                                      role_attention_mask: Optional[torch.Tensor] = None,
                                      role_token_type_ids: Optional[torch.Tensor] = None,
                                      bert_outputs: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None
                                      ) -> torch.FloatTensor:
        role_input_ids = self.convert_triplet_batch(role_input_ids)
        role_attention_mask = self.convert_triplet_batch(role_attention_mask)
        if role_token_type_ids is not None:
            role_token_type_ids = self.convert_triplet_batch(role_token_type_ids)

        kge_outputs = self.kge_model(input_ids=role_input_ids,
                                     attention_mask=role_attention_mask,
                                     token_type_ids=role_token_type_ids,
                                     return_dict=True)
        role_output = kge_outputs.pooler_output

        sequence_output = bert_outputs.last_hidden_state
        trigger_output = torch.zeros(sequence_output.shape[0], sequence_output.shape[2]).to(self.bert.device)

        neg_trigger_output = torch.zeros(sequence_output.shape[0], sequence_output.shape[2]).to(self.bert.device) \
            if neg_trigger_mask is not None else None
        argument_output = torch.zeros(sequence_output.shape[0], sequence_output.shape[2]).to(self.bert.device)

        for i in range(sequence_output.shape[0]):
            trigger_output[i, :] = sequence_output[i, trigger_mask[i]].mean(dim=0)
            if neg_trigger_output is not None:
                neg_trigger_output[i, :] = sequence_output[i, neg_trigger_mask[i]].mean(dim=0)
            argument_output[i, :] = sequence_output[i, argument_mask[i]].mean(dim=0)

        extend_num = role_output.shape[0] // argument_output.shape[0]
        argument_output = self.extend_triplet_batch(argument_output, extend_num)
        if neg_trigger_output is not None:
            trigger_output = self.extend_trigger_batch(trigger_output, extend_num - 1, neg_trigger_output)
        else:
            trigger_output = self.extend_triplet_batch(trigger_output, extend_num)

        kge_score = self.compute_kge_score(argument_output, trigger_output, role_output)

        return kge_score

    def compute_kge_score(self,
                          argument_output: torch.FloatTensor,
                          trigger_output: torch.FloatTensor,
                          role_output: torch.FloatTensor
                          ) -> torch.FloatTensor:
        if self.triplet_comb == "ar_t":
            kge_score = self.kge_scorer(argument_output, trigger_output, role_output)
        elif self.triplet_comb == "at_r":
            kge_score = self.kge_scorer(argument_output, role_output, trigger_output)
        elif self.triplet_comb == "tr_a":
            kge_score = self.kge_scorer(trigger_output, argument_output, trigger_output)
        else:
            raise ValueError(f"the triplet combination called {self.triplet_comb} is not defined")

        return kge_score

    @staticmethod
    def convert_triplet_batch(role_input: torch.Tensor) -> torch.Tensor:
        # (batch_size, sample_num, sequence_length)
        # -> (batch_size * sample_num, sequence_length)
        role_converted = role_input.view((-1, role_input.shape[-1]))

        return role_converted

    @staticmethod
    def extend_triplet_batch(batch_input: torch.Tensor,
                             extend_num: int
                             ) -> torch.Tensor:
        # (batch_size, hidden_dim)
        # -> (batch_size, sample_num, hidden_dim)
        batch_extended = torch.stack([batch_input] * extend_num, dim=1)
        # (batch_size, sample_num, hidden_dim)
        # -> (batch_size * sample_num, hidden_dim)
        batch_extended = batch_extended.view((-1, batch_extended.shape[-1]))

        return batch_extended

    @staticmethod
    def extend_trigger_batch(batch_input: torch.Tensor,
                             extend_num: int,
                             neg_trigger_output: torch.Tensor,
                             ) -> torch.Tensor:
        # (batch_size, hidden_dim)
        # -> (batch_size, sample_num, hidden_dim)
        batch_extended = torch.stack([batch_input] * extend_num + [neg_trigger_output], dim=1)
        # (batch_size, sample_num, hidden_dim)
        # -> (batch_size * sample_num, hidden_dim)
        batch_extended = batch_extended.view((-1, batch_extended.shape[-1]))

        return batch_extended

    def predict_crf(self,
                    input_ids: Optional[torch.Tensor] = None,
                    token_type_ids: Optional[torch.Tensor] = None,
                    attention_mask: Optional[torch.Tensor] = None
                    ) -> List[List[int]]:
        logits = self.identification_module_forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )[0]
        pred_label = self.crf.decode(emissions=logits, mask=attention_mask.bool())

        return pred_label

    def predict_kge(self,
                    input_ids: Optional[torch.Tensor] = None,
                    token_type_ids: Optional[torch.Tensor] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    trigger_mask: Optional[torch.Tensor] = None,
                    argument_mask: Optional[torch.Tensor] = None,
                    role_input_ids: Optional[torch.Tensor] = None,
                    role_attention_mask: Optional[torch.Tensor] = None,
                    role_token_type_ids: Optional[torch.Tensor] = None
                    ) -> torch.FloatTensor:
        bert_outputs = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 return_dict=True)
        kge_score = self.classification_module_forward(
            trigger_mask=trigger_mask,
            argument_mask=argument_mask,
            role_input_ids=role_input_ids,
            role_attention_mask=role_attention_mask,
            role_token_type_ids=role_token_type_ids,
            bert_outputs=bert_outputs
        )
        pred_probs = self.sigmoid(kge_score)

        return pred_probs

    def init_kge_encoder_from_bert(self) -> None:
        kge_encoder_dict = self.kge_model.encoder.state_dict()
        bert_encoder_dict = self.bert.encoder.state_dict()
        match_dict = {k: v for k, v in bert_encoder_dict.items() if k in kge_encoder_dict}
        kge_encoder_dict.update(match_dict)
        self.kge_model.encoder.load_state_dict(kge_encoder_dict.copy())


class RobertaCrfKge(RobertaPreTrainedModel):
    def __init__(self,
                 config: Optional[RobertaConfig] = None,
                 need_lstm: Optional[bool] = True,
                 lstm_dim: Optional[int] = 128,
                 num_lstm_layer: Optional[int] = 1,
                 num_role_encoder_layers: Optional[int] = 2,
                 kge_scorer_name: Optional[AnyStr] = "TransE",
                 triplet_comb: Optional[AnyStr] = "ar_t"
                 ) -> None:
        super(RobertaCrfKge, self).__init__(config)

        self.need_lstm = need_lstm
        self.lstm_dim = lstm_dim
        self.num_lstm_layer = num_lstm_layer
        self.num_role_encoder_layers = num_role_encoder_layers
        self.kge_scorer_name = kge_scorer_name
        self.triplet_comb = triplet_comb

        # argument identification module
        self.roberta = RobertaModel(config=config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        out_dim = config.hidden_size

        self.lstm = None
        if self.need_lstm:
            out_dim = self.lstm_dim * 2
            self.lstm = nn.LSTM(input_size=config.hidden_size,
                                hidden_size=self.lstm_dim,
                                num_layers=self.num_lstm_layer,
                                bidirectional=True,
                                batch_first=True)

        self.classifier = nn.Linear(in_features=out_dim, out_features=config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        # argument classification module
        kge_config = deepcopy(config)
        kge_config.num_hidden_layers = self.num_role_encoder_layers
        self.kge_model = RobertaModel(config=kge_config)
        self.kge_model.embeddings = self.roberta.embeddings
        self.sigmoid = nn.Sigmoid()
        self.kge_criterion = nn.BCEWithLogitsLoss()
        self.mse_criterion = nn.MSELoss()

        self.gamma = config.hidden_size // 2
        self.kge_scorer = None
        if self.kge_scorer_name == "TransE":
            self.kge_scorer = TransE(self.gamma)
        elif self.kge_scorer_name == "DistMult":
            self.kge_scorer = DistMult()
        elif self.kge_scorer_name == "ComplEx":
            self.kge_scorer = ComplEx()
        elif self.kge_scorer_name == "RotatE":
            self.kge_scorer = RotatE(self.gamma, config.hidden_size)
        else:
            raise ValueError(f"no module called {self.kge_scorer_name}")

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                trigger_mask: Optional[torch.Tensor] = None,
                neg_trigger_mask: Optional[torch.Tensor] = None,
                argument_mask: Optional[torch.Tensor] = None,
                role_input_ids: Optional[torch.Tensor] = None,
                role_attention_mask: Optional[torch.Tensor] = None,
                role_token_type_ids: Optional[torch.Tensor] = None,
                triplet_label: Optional[torch.Tensor] = None
                ) -> DistarOutput:
        # argument identification module forward
        logits, bert_outputs = self.identification_module_forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        loss = None
        if labels is not None:
            loss = -1.0 * self.crf(emissions=logits,
                                   tags=labels,
                                   mask=attention_mask.bool())
            loss = loss / input_ids.shape[0]
            
        # argument classification module forward
        kge_loss = None
        contra_loss = None
        kge_score = None
        if role_input_ids is not None:
            kge_score, contra_loss = self.classification_module_forward(
                trigger_mask=trigger_mask,
                neg_trigger_mask=neg_trigger_mask,
                argument_mask=argument_mask,
                role_input_ids=role_input_ids,
                role_attention_mask=role_attention_mask,
                role_token_type_ids=role_token_type_ids,
                bert_outputs=bert_outputs,
                has_pos_role=False
            )
            triplet_label = triplet_label.view((-1,))
            kge_loss = self.kge_criterion(kge_score, triplet_label)
            loss += kge_loss
            if contra_loss is not None:
                loss += contra_loss

        return DistarOutput(
            loss=loss,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
            logits=logits,
            kge_loss=kge_loss,
            contra_loss=contra_loss,
            kge_probs=self.sigmoid(kge_score) if kge_score is not None else None,
        )

    def identification_module_forward(self,
                                      input_ids: Optional[torch.Tensor] = None,
                                      token_type_ids: Optional[torch.Tensor] = None,
                                      attention_mask: Optional[torch.Tensor] = None
                                      ) -> Tuple[torch.FloatTensor,
                                                 BaseModelOutputWithPoolingAndCrossAttentions]:
        bert_outputs = self.roberta(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    return_dict=True)
        sequence_output = bert_outputs.last_hidden_state
        
        if self.need_lstm:
            sequence_output = self.lstm(sequence_output)[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits, bert_outputs

    def classification_module_forward(self,
                                      trigger_mask: Optional[torch.Tensor] = None,
                                      neg_trigger_mask: Optional[torch.Tensor] = None,
                                      argument_mask: Optional[torch.Tensor] = None,
                                      role_input_ids: Optional[torch.Tensor] = None,
                                      role_attention_mask: Optional[torch.Tensor] = None,
                                      role_token_type_ids: Optional[torch.Tensor] = None,
                                      bert_outputs: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None,
                                      has_pos_role: Optional[bool] = False,
                                      ) -> Union[torch.FloatTensor, Tuple]:
        role_input_ids = self.convert_triplet_batch(role_input_ids)
        role_attention_mask = self.convert_triplet_batch(role_attention_mask)
        if role_token_type_ids is not None:
            role_token_type_ids = self.convert_triplet_batch(role_token_type_ids)

        kge_outputs = self.kge_model(input_ids=role_input_ids,
                                     attention_mask=role_attention_mask,
                                     token_type_ids=role_token_type_ids,
                                     return_dict=True)
        role_output = kge_outputs.pooler_output

        sequence_output = bert_outputs.last_hidden_state
        trigger_output = torch.zeros(sequence_output.shape[0], sequence_output.shape[2]).to(self.roberta.device)
        neg_trigger_output = torch.zeros(sequence_output.shape[0], sequence_output.shape[2]).to(self.roberta.device) \
            if neg_trigger_mask is not None else None
        argument_output = torch.zeros(sequence_output.shape[0], sequence_output.shape[2]).to(self.roberta.device)

        for i in range(sequence_output.shape[0]):
            trigger_output[i, :] = sequence_output[i, trigger_mask[i]].mean(dim=0)
            if neg_trigger_output is not None:
                neg_trigger_output[i, :] = sequence_output[i, neg_trigger_mask[i]].mean(dim=0)
            argument_output[i, :] = sequence_output[i, argument_mask[i]].mean(dim=0)

        extend_num = role_output.shape[0] // argument_output.shape[0]
        argument_output = self.extend_triplet_batch(argument_output, extend_num)
        if neg_trigger_output is not None:
            trigger_output = self.extend_trigger_batch(trigger_output, extend_num - 1, neg_trigger_output)
        else:
            trigger_output = self.extend_triplet_batch(trigger_output, extend_num)

        kge_score = self.compute_kge_score(argument_output, trigger_output, role_output)

        if has_pos_role:
            triplet_num = role_output.shape[0] // extend_num
            a_role_output = role_output[[extend_num * i for i in range(triplet_num)]]
            p_role_output = role_output[[extend_num * i + (extend_num - 1) for i in range(triplet_num)]]
            n_role_output = role_output[[n for i in range(triplet_num)
                                         for n in range(extend_num * i + 1, extend_num * (i + 1) - 1)]]

            a_role_output = self.extend_triplet_batch(a_role_output, n_role_output.shape[0] // a_role_output.shape[0])
            p_role_output = self.extend_triplet_batch(p_role_output, n_role_output.shape[0] // p_role_output.shape[0])

            an_dist = torch.sqrt(self.mse_criterion(a_role_output, n_role_output)).sum()
            ap_dist = torch.sqrt(self.mse_criterion(a_role_output, p_role_output)).sum()
            contra_loss = torch.max(torch.tensor(0), ap_dist - an_dist + 0.2)

            return kge_score, contra_loss
        else:
            return kge_score, None

    def compute_kge_score(self,
                          argument_output: torch.FloatTensor,
                          trigger_output: torch.FloatTensor,
                          role_output: torch.FloatTensor
                          ) -> torch.FloatTensor:
        if self.triplet_comb == "ar_t":
            kge_score = self.kge_scorer(argument_output, trigger_output, role_output)
        elif self.triplet_comb == "at_r":
            kge_score = self.kge_scorer(argument_output, role_output, trigger_output)
        elif self.triplet_comb == "tr_a":
            kge_score = self.kge_scorer(trigger_output, argument_output, trigger_output)
        else:
            raise ValueError(f"the triplet combination called {self.triplet_comb} is not defined")

        return kge_score

    @staticmethod
    def convert_triplet_batch(role_input: torch.Tensor) -> torch.Tensor:
        # (batch_size, sample_num, sequence_length)
        # -> (batch_size * sample_num, sequence_length)
        role_converted = role_input.view((-1, role_input.shape[-1]))

        return role_converted

    @staticmethod
    def extend_triplet_batch(batch_input: torch.Tensor,
                             extend_num: int
                             ) -> torch.Tensor:
        # (batch_size, hidden_dim)
        # -> (batch_size, sample_num, hidden_dim)
        batch_extended = torch.stack([batch_input] * extend_num, dim=1)
        # (batch_size, sample_num, hidden_dim)
        # -> (batch_size * sample_num, hidden_dim)
        batch_extended = batch_extended.view((-1, batch_extended.shape[-1]))

        return batch_extended

    @staticmethod
    def extend_trigger_batch(batch_input: torch.Tensor,
                             extend_num: int,
                             neg_trigger_output: torch.Tensor,
                             ) -> torch.Tensor:
        # (batch_size, hidden_dim)
        # -> (batch_size, sample_num, hidden_dim)
        batch_extended = torch.stack([batch_input] * extend_num + [neg_trigger_output], dim=1)
        # (batch_size, sample_num, hidden_dim)
        # -> (batch_size * sample_num, hidden_dim)
        batch_extended = batch_extended.view((-1, batch_extended.shape[-1]))

        return batch_extended

    def predict_crf(self,
                    input_ids: Optional[torch.Tensor] = None,
                    token_type_ids: Optional[torch.Tensor] = None,
                    attention_mask: Optional[torch.Tensor] = None
                    ) -> List[List[int]]:
        logits = self.identification_module_forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )[0]
        pred_label = self.crf.decode(emissions=logits, mask=attention_mask.bool())

        return pred_label

    def predict_kge(self,
                    input_ids: Optional[torch.Tensor] = None,
                    token_type_ids: Optional[torch.Tensor] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    trigger_mask: Optional[torch.Tensor] = None,
                    argument_mask: Optional[torch.Tensor] = None,
                    role_input_ids: Optional[torch.Tensor] = None,
                    role_attention_mask: Optional[torch.Tensor] = None,
                    role_token_type_ids: Optional[torch.Tensor] = None,
                    has_pos_role: Optional[bool] = False,
                    ) -> torch.FloatTensor:
        bert_outputs = self.roberta(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    return_dict=True)
        kge_score, contra_loss = self.classification_module_forward(
            trigger_mask=trigger_mask,
            argument_mask=argument_mask,
            role_input_ids=role_input_ids,
            role_attention_mask=role_attention_mask,
            role_token_type_ids=role_token_type_ids,
            bert_outputs=bert_outputs,
            has_pos_role=has_pos_role
        )
        pred_probs = self.sigmoid(kge_score)

        return pred_probs

    def init_kge_encoder_from_bert(self) -> None:
        kge_encoder_dict = self.kge_model.encoder.state_dict()
        bert_encoder_dict = self.roberta.encoder.state_dict()
        match_dict = {k: v for k, v in bert_encoder_dict.items() if k in kge_encoder_dict}
        kge_encoder_dict.update(match_dict)
        self.kge_model.encoder.load_state_dict(kge_encoder_dict.copy())


class DistarModel(object):
    @staticmethod
    def load_model(model_name_or_path: AnyStr,
                   num_labels: Optional[int] = 3,
                   need_lstm: Optional[bool] = True,
                   lstm_dim: Optional[int] = 128,
                   num_lstm_layer: Optional[int] = 1,
                   num_role_encoder_layers: Optional[int] = 2,
                   kge_scorer_name: Optional[AnyStr] = "TransE",
                   triplet_comb: Optional[AnyStr] = "ar_t",
                   trigger_left_token: Optional[AnyStr] = "[TRI]",
                   trigger_right_token: Optional[AnyStr] = "[TRI]"
                   ) -> Tuple[Union[BertTokenizerFast, RobertaTokenizerFast],
                              Union[BertConfig, RobertaConfig],
                              Union[BertCrfKge, RobertaCrfKge]]:
        if "bert-" in model_name_or_path:
            config = BertConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
            model = BertCrfKge.from_pretrained(
                model_name_or_path,
                config=config,
                need_lstm=need_lstm,
                lstm_dim=lstm_dim,
                num_lstm_layer=num_lstm_layer,
                num_role_encoder_layers=num_role_encoder_layers,
                kge_scorer_name=kge_scorer_name,
                triplet_comb=triplet_comb
            )

            tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
            add_vocab_num = tokenizer.add_tokens([trigger_left_token, trigger_right_token])
            if add_vocab_num > 0:
                model.bert.resize_token_embeddings(len(tokenizer))
                print(f"\nresize token embeddings to {len(tokenizer)} "
                      f"({len(tokenizer) - add_vocab_num} + {add_vocab_num})\n")

        elif "roberta-" in model_name_or_path:
            config = RobertaConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
            model = RobertaCrfKge.from_pretrained(
                model_name_or_path,
                config=config,
                need_lstm=need_lstm,
                lstm_dim=lstm_dim,
                num_lstm_layer=num_lstm_layer,
                num_role_encoder_layers=num_role_encoder_layers,
                kge_scorer_name=kge_scorer_name,
                triplet_comb=triplet_comb
            )

            tokenizer = RobertaTokenizerFast.from_pretrained(model_name_or_path)
            add_vocab_num = tokenizer.add_tokens([trigger_left_token, trigger_right_token])
            if add_vocab_num > 0:
                model.roberta.resize_token_embeddings(len(tokenizer))
                print(f"\nresize token embeddings to {len(tokenizer)} "
                      f"({len(tokenizer) - add_vocab_num} + {add_vocab_num})\n")
        else:
            raise ValueError(f"model type {model_name_or_path} not matched")

        return tokenizer, config, model
