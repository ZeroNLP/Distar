import copy
import random
from typing import List, Dict, AnyStr, Any, Union, Optional, Tuple

import numpy as np
from nltk import word_tokenize, pos_tag

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import BertTokenizerFast, RobertaTokenizerFast


class DistarDataset(Dataset):
    def __init__(self,
                 data: List[Dict[AnyStr, Any]],
                 tokenizer: Union[BertTokenizerFast, RobertaTokenizerFast],
                 role_desc: Dict[AnyStr, Dict[AnyStr, AnyStr]],
                 tag2index: Dict[AnyStr, int],
                 index2tag: Dict[int, AnyStr],
                 trigger_left_token: Optional[AnyStr] = "[TRI]",
                 trigger_right_token: Optional[AnyStr] = "[TRI]",
                 max_input_length: int = 128,
                 max_role_length: int = 32,
                 num_neg_role: int = 1,
                 split_types: Dict[AnyStr, Any] = None
                 ) -> None:
        super(DistarDataset, self).__init__()

        self.tokenizer = tokenizer
        self.role_desc = role_desc
        self.tag2index = tag2index
        self.index2tag = index2tag
        self.role_dict = {event_type: role.keys() for event_type, role in role_desc.items()}
        self.trigger_left_token = trigger_left_token
        self.trigger_right_token = trigger_right_token
        self.max_input_length = max_input_length
        self.max_role_length = max_role_length
        self.num_neg_role = num_neg_role
        self.split_types = split_types
        # self.seen_roles = set(
        #     [role
        #      for event_type, roles in self.role_dict.items()
        #      for role in roles
        #      if event_type in self.split_types["seen_types"]]
        # )
        # # print(self.seen_roles)
        # self.role2type = {role: [] for role in self.seen_roles}
        # for event_type, roles in self.role_dict.items():
        #     if event_type in self.split_types["seen_types"]:
        #         for role in roles:
        #             self.role2type[role].append(event_type)
        # print(self.role2type)

        self.skip_num = 0
        self.features = self.generate_features(data)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]

    @staticmethod
    def generate_neg_trigger_data(data: List[Dict[AnyStr, Any]]
                                  ) -> List[Dict[AnyStr, Any]]:
        neg_trigger_data = []
        for sample in tqdm(data, desc="generate neg trigger"):
            sample_neg = copy.deepcopy(sample)

            sentence = sample_neg["sentence"]
            tagged_words = pos_tag(word_tokenize(sentence))
            words_with_indices = []
            idx = 0
            for word, pos in tagged_words:
                start_idx = sentence.find(word, idx)
                end_idx = start_idx + len(word)

                if word != sample["triplet"]["trigger"]["text"]:
                    words_with_indices.append((word, start_idx, end_idx, pos))
                idx = end_idx

            verbs_with_indices = [item for item in words_with_indices if item[-1].startswith("VB")]

            # print(sentence)
            # print(verbs_with_indices)
            # print(sample["triplet"]["trigger"])
            # break

            # if len(verbs_with_indices) == 0:
            #     nouns_with_indices = [item for item in words_with_indices if item[-1].startswith("NN")]
            #     if len(nouns_with_indices) > 0:
            #         verbs_with_indices.append(random.choice(nouns_with_indices))
            #     else:
            #         verbs_with_indices.append(random.choice(words_with_indices))
            if len(verbs_with_indices) > 0:
                neg_trigger = random.choice(verbs_with_indices)
                sample_neg["triplet"]["label"] = 0
                sample_neg["triplet"]["trigger"]["text"] = neg_trigger[0]
                sample_neg["triplet"]["trigger"]["start"] = neg_trigger[1]
                sample_neg["triplet"]["trigger"]["end"] = neg_trigger[2]
                neg_trigger_data.append(sample_neg)

        random.shuffle(neg_trigger_data)
        data = data + neg_trigger_data[:500]
        random.shuffle(data)

        return data

    @staticmethod
    def generate_neg_trigger(sentence: AnyStr,
                             trigger: Dict[AnyStr, Any]
                             ) -> Union[Dict[AnyStr, Any], None]:
        tagged_words = pos_tag(word_tokenize(sentence))
        words_with_indices = []
        idx = 0
        for word, pos in tagged_words:
            start_idx = sentence.find(word, idx)
            end_idx = start_idx + len(word)

            if word != trigger["text"]:
                words_with_indices.append((word, start_idx, end_idx, pos))
                # if sentence[start_idx:end_idx] != word:
                #     print(sentence[start_idx: end_idx], word)
            idx = end_idx

        verbs_with_indices = [item for item in words_with_indices if item[-1].startswith("VB")]
        if len(verbs_with_indices) == 0:
            verbs_with_indices = words_with_indices

        neg_trigger_item = random.choice(verbs_with_indices)
        neg_trigger = {
            "text": neg_trigger_item[0],
            "start": neg_trigger_item[1],
            "end": neg_trigger_item[2]
        }
        return neg_trigger

    def generate_features(self,
                          data: List[Dict[AnyStr, Any]]
                          ) -> List[Dict[AnyStr, Any]]:
        features = []

        for sample in tqdm(data, desc="generate features"):
            feature = {}

            sentence = sample["sentence"]
            event_type = sample["event_type"]
            trigger = sample["trigger"]
            arguments = sample["arguments"]
            triplet = sample["triplet"]

            # add trigger position token
            trigger_text = sentence[trigger["start"]: trigger["end"]]
            sentence_marked = sentence[:trigger["start"]] \
                       + self.trigger_left_token \
                       + trigger_text \
                       + self.trigger_right_token \
                       + sentence[trigger["end"]:]

            # encode the input text
            prefix_len = len(trigger["text"] + self.tokenizer.sep_token)
            input_text = trigger["text"] + self.tokenizer.sep_token + sentence_marked
            input_text_encoded = self.tokenizer(
                input_text,
                truncation=True,
                max_length=self.max_input_length,
                padding="max_length",
                return_offsets_mapping=True
            )

            # get start mapping and end mapping
            offset_mapping = input_text_encoded.pop("offset_mapping")
            start_mapping = {offset[0]: index
                             for index, offset in enumerate(offset_mapping)
                             if offset}
            end_mapping = {offset[-1]: index
                           for index, offset in enumerate(offset_mapping)
                           if offset}

            # generate label for arguments
            label = [self.tag2index["O"]] * len(input_text_encoded["input_ids"])

            for argument in arguments:
                start_index, end_index = self.get_argument_offset(trigger, argument, prefix_len)

                if (start_index not in start_mapping.keys()) or (end_index not in end_mapping.keys()):
                    self.skip_num += 1
                    continue

                start_token_index = start_mapping[start_index]
                end_token_index = end_mapping[end_index]
                label[start_token_index] = self.tag2index["B"]
                for i in range(start_token_index + 1, end_token_index + 1):
                    label[i] = self.tag2index["I"]

            # generate feature of input text
            feature["labels"] = torch.tensor(label).long()
            feature["input_ids"] = torch.tensor(input_text_encoded["input_ids"]).long()
            feature["attention_mask"] = torch.tensor(input_text_encoded["attention_mask"]).long()
            if "token_type_ids" in input_text_encoded.keys():
                feature["token_type_ids"] = torch.tensor(input_text_encoded["token_type_ids"]).long()

            # generate trigger mask
            triplet_trigger = triplet["trigger"]
            trigger_mask = torch.zeros_like(feature["input_ids"]).bool()

            # trigger_start = triplet_trigger["start"] + prefix_len + len(self.trigger_left_token)
            # trigger_end = triplet_trigger["end"] + prefix_len + len(self.trigger_right_token)
            trigger_start, trigger_end = self.get_argument_offset(trigger, triplet_trigger, prefix_len)
            if (trigger_start not in start_mapping.keys()) or (trigger_end not in end_mapping.keys()):
                self.skip_num += 1
                continue

            trigger_token_start = start_mapping[trigger_start]
            trigger_token_end = end_mapping[trigger_end]

            if trigger_token_end == trigger_token_end:
                trigger_mask[trigger_token_start] = True
            else:
                trigger_mask[trigger_token_start: trigger_token_end] = True

            # neg_trigger = self.generate_neg_trigger(sentence, trigger)
            # neg_trigger_mask = torch.zeros_like(feature["input_ids"]).bool()
            # neg_trigger_start, neg_trigger_end = self.get_argument_offset(trigger, neg_trigger, prefix_len)
            # if (neg_trigger_start not in start_mapping.keys()) or (neg_trigger_end not in end_mapping.keys()):
            #     neg_trigger_token_start = random.choice(list(start_mapping.values()))
            #     neg_trigger_token_end = neg_trigger_token_start + 1
            # else:
            #     neg_trigger_token_start = start_mapping[neg_trigger_start]
            #     neg_trigger_token_end = end_mapping[neg_trigger_end]
            # if neg_trigger_token_end == neg_trigger_token_end:
            #     neg_trigger_mask[neg_trigger_token_start] = True
            # else:
            #     neg_trigger_mask[neg_trigger_token_start: neg_trigger_token_end] = True
            # feature["neg_trigger_mask"] = neg_trigger_mask

            # generate argument mask
            triplet_argument = triplet["argument"]
            argument_mask = torch.zeros_like(feature["input_ids"]).bool()
            argument_start, argument_end = self.get_argument_offset(trigger, triplet_argument, prefix_len)

            if (argument_start not in start_mapping.keys()) or (argument_end not in end_mapping.keys()):
                self.skip_num += 1
                continue

            argument_token_start = start_mapping[argument_start]
            argument_token_end = end_mapping[argument_end]

            if argument_token_start == argument_token_end:
                argument_mask[argument_token_start] = True
            else:
                argument_mask[argument_token_start: argument_token_end] = True

            # generate positive and negative role text
            role_text_list = []

            if "role_description" in triplet.keys():
                pos_role_text = triplet["role_description"]
            else:
                if triplet["role"] + "_Arg" not in self.role_desc[event_type].keys():
                    print(sentence)
                    print(triplet["role"])
                pos_role_text = self.role_desc[event_type][triplet["role"] + "_Arg"]
            role_text_list.append(pos_role_text)

            
            neg_roles = sorted(list(self.role_dict[event_type] - [triplet["role"] + "_Arg"]))
            # neg_roles = sorted(list(self.seen_roles - {triplet["role"] + "_Arg"}))
            for neg_role in np.random.choice(neg_roles, self.num_neg_role, replace=True):
                neg_role_text = self.role_desc[event_type][neg_role]
                # neg_role_text = self.role_desc[np.random.choice(self.role2type[neg_role], 1)[0]][neg_role]
                role_text_list.append(neg_role_text)

            if "neg_trigger_mask" in feature.keys():
                role_text_list.append(pos_role_text)

            if "pos_role" in triplet.keys():
                role_text_list.append(triplet["pos_role_description"])

            role_encoded = self.tokenizer(
                role_text_list,
                truncation=True,
                max_length=self.max_role_length,
                padding="max_length",
                return_offsets_mapping=False
            )

            # generate triplet label
            triplet_label = [triplet["label"]] + [0] * self.num_neg_role
            if "neg_trigger_mask" in feature.keys():
                triplet_label.append(0)
            if "pos_role" in triplet.keys():
                triplet_label.append(1)

            # generate triplet feature
            feature["trigger_mask"] = trigger_mask
            feature["argument_mask"] = argument_mask
            feature["role_input_ids"] = torch.tensor(role_encoded["input_ids"]).long()
            feature["role_attention_mask"] = torch.tensor(role_encoded["attention_mask"]).long()
            if "token_type_ids" in role_encoded.keys():
                feature["role_token_type_ids"] = torch.tensor(role_encoded["token_type_ids"]).long()
            feature["triplet_label"] = torch.tensor([triplet_label]).float()

            features.append(feature)

        print(f"skip num: {self.skip_num}")
        print(f"generate {len(features)} features")

        return features

    def get_argument_offset(self,
                            trigger: Dict[AnyStr, Union[AnyStr, int]],
                            argument: Dict[AnyStr, Union[AnyStr, int]],
                            prefix_len: int
                            ) -> Tuple[int, int]:
        start_index = self.move_index(argument["start"], trigger, prefix_len)
        end_index = self.move_index(argument["end"], trigger, prefix_len)

        return start_index, end_index

    def move_index(self,
                   index: int,
                   trigger: Dict[AnyStr, Union[AnyStr, int]],
                   prefix_len: int
                   ) -> int:
        if index < trigger["start"]:
            index += prefix_len
        elif index <= trigger["end"]:
            index += prefix_len + len(self.trigger_left_token)
        else:
            index += prefix_len \
                     + len(self.trigger_left_token) \
                     + len(self.trigger_right_token)

        return index
