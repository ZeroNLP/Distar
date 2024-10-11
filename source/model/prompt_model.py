from typing import Optional
from dataclasses import dataclass

import torch
from transformers import RobertaForMaskedLM
from transformers.modeling_outputs import ModelOutput


@dataclass
class PromptOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class RobertaForPrompt(RobertaForMaskedLM):
    def __init__(self, config, ap_label_space, type_label_space, mask_token_id=50264):
        super(RobertaForPrompt, self).__init__(config)

        self.config = config
        self.neutral_label = [ap_label_space[x] for x in ap_label_space.keys() if "neutral" in x][0]
        self.ap_label_space = ap_label_space
        self.type_label_space = type_label_space
        self.ap_vocab_mask = None
        self.ap_vocab_mask_no_neutral = None
        self.type_vocab_mask = None
        self.mask_token_id = mask_token_id

    def forward(self, input_ids, attention_mask=None, label_ap_idx=None, label_type_idx=None, labels=None):
        outputs = super(RobertaForPrompt, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        mask_ids = (input_ids == self.mask_token_id).to(self.roberta.device)
        return PromptOutput(
            loss=outputs.loss,
            logits=outputs.logits[mask_ids, :].view(input_ids.shape[0], 2, -1)
        )

    def get_ap_vocab_mask(self):
        if self.ap_vocab_mask is not None:
            return self.ap_vocab_mask

        self.ap_vocab_mask = torch.zeros(self.config.vocab_size).bool()
        for label in self.ap_label_space.values():
            self.ap_vocab_mask[label] = True
        self.ap_vocab_mask = self.ap_vocab_mask.to(self.roberta.device)

        self.ap_vocab_mask_no_neutral = self.ap_vocab_mask.clone()
        self.ap_vocab_mask_no_neutral[self.neutral_label] = False

        return self.ap_vocab_mask

    def get_type_vocab_mask(self):
        if self.type_vocab_mask is not None:
            return self.type_vocab_mask

        self.type_vocab_mask = torch.zeros(self.config.vocab_size).bool()
        for label in self.type_label_space.values():
            self.type_vocab_mask[label] = True
        self.type_vocab_mask = self.type_vocab_mask.to(self.roberta.device)

        return self.type_vocab_mask

    def predict(self, input_ids, attention_mask=None):
        outputs = self.forward(input_ids, attention_mask)
        logits = outputs.logits
        ap_logits = logits[:, 0, :]
        type_logits = logits[:, 1, :]

        if self.ap_vocab_mask_no_neutral is None:
            _ = self.get_ap_vocab_mask()
        self.type_vocab_mask = self.get_type_vocab_mask()

        ap_logits = ap_logits[:, self.ap_vocab_mask]
        type_logits = type_logits[:, self.type_vocab_mask]

        ap_probs = ap_logits.softmax(dim=-1)
        type_probs = type_logits.softmax(dim=-1)
        prob_vectors = torch.cat([ap_probs[:, :1], ap_probs[:, 2:], type_probs], dim=1)

        return prob_vectors
