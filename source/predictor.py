import json
import os
from copy import deepcopy
from argparse import Namespace
from typing import Dict, AnyStr, Union, Tuple, List, Optional

from tqdm.auto import tqdm

import torch
from transformers import (BertConfig,
                          RobertaConfig,
                          BertTokenizerFast,
                          RobertaTokenizerFast,
                          BatchEncoding,
                          AutoTokenizer)

from .utils.logger import logger
from .utils.utils import save_json_data
from .model.model import BertCrfKge, RobertaCrfKge
from .model.prompt_model import RobertaForPrompt


class DistarPredictor(object):
    def __init__(self,
                 args: Namespace,
                 role_desc: Dict[AnyStr, Dict[AnyStr, AnyStr]],
                 tokenizer: Union[BertTokenizerFast, RobertaTokenizerFast],
                 config: Union[BertConfig, RobertaConfig],
                 model: Union[BertCrfKge, RobertaCrfKge],
                 output_dir: AnyStr
                 ) -> None:
        self.args = args
        self.role_desc = role_desc
        self.device = args.device
        self.tokenizer = tokenizer
        self.config = config
        self.model = model
        self.trigger_left_token = args.trigger_left_token
        self.trigger_right_token = args.trigger_right_token

        self.model.eval()
        self.prediction_path = os.path.join(output_dir, "prediction")
        if not os.path.exists(self.prediction_path):
            os.mkdir(self.prediction_path)

        # --------------------------------------------------------------
        # with open("/home/zdlu/project/trigger_classification/onto_data2/ap_label_space.json", "r") as f:
        #     self.ap_label_space = json.load(f)
        # with open("/home/zdlu/project/trigger_classification/onto_data2/type_label_space.json", "r") as f:
        #     self.type_label_space = json.load(f)
        # with open("/home/zdlu/project/trigger_classification/onto_data2/role_type_vector.json", "r") as f:
        #     self.role_type_vector = json.load(f)
        #     for event_type in self.role_type_vector.keys():
        #         for role in self.role_type_vector[event_type].keys():
        #             self.role_type_vector[event_type][role] = torch.tensor(
        #                 [self.role_type_vector[event_type][role]]
        #             ).to(self.device)
        #
        # prompt_model_path = "/data/zdlu/prompt_ontology_output/roberta-base-prompt-ap-type_sup/checkpoint-1950"
        # self.prompt_tokenizer = AutoTokenizer.from_pretrained(prompt_model_path)
        # self.prompt_model = RobertaForPrompt.from_pretrained(
        #     prompt_model_path,
        #     ap_label_space=self.ap_label_space,
        #     type_label_space=self.type_label_space
        # ).to(self.device)
        # self.prompt_model.eval()
        # self.prompt_template = 'The argument "{argument}" is a/an {ap_label} {type_label} entity. '
        # --------------------------------------------------------------

        print()
        logger.info("Distar Predictor")
        logger.info(f"{os.path.abspath(self.prediction_path)}\n")

    def prepare_prompt_input_ids(self, sentence: AnyStr, argument: AnyStr) -> torch.Tensor:
        prompt_text = self.prompt_template.format(
            argument=argument,
            ap_label=self.prompt_tokenizer.mask_token,
            type_label=self.prompt_tokenizer.mask_token
        )
        max_sent_length = 192
        input_ids = [self.prompt_tokenizer.cls_token_id]
        sentence_ids = self.prompt_tokenizer.encode(sentence, add_special_tokens=False)
        if len(sentence_ids) > max_sent_length:
            input_ids += sentence_ids[:max_sent_length]
        else:
            input_ids += sentence_ids
        input_ids += [self.prompt_tokenizer.sep_token_id]

        prompt_ids = self.prompt_tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids += prompt_ids + [self.prompt_tokenizer.sep_token_id]
        input_ids = torch.tensor([input_ids]).to(self.device)

        return input_ids

    def predict_instances(self,
                          data: List[Dict[AnyStr, Union[Dict, AnyStr, int]]],
                          trigger_type: Optional[AnyStr] = "gold"
                          ) -> List[Dict[AnyStr, Union[Dict, AnyStr, int, float]]]:
        data = deepcopy(data)

        if trigger_type == "gold":
            data = self.take_gold_trigger_as_input(data)

        if "pred_event_mentions" not in data[0].keys():
            raise ValueError("no 'pred_event_mentions' field in data, "
                             "the 'pred_event_mentions' field needs to "
                             "contain predicted trigger and event type")

        for instance in tqdm(data, desc="Predicting"):
            sentence = instance["sentence"]

            for event in instance["pred_event_mentions"]:
                event_type = event["event_type"]
                trigger = event["trigger"]

                pred_result = self.predict_one_instance(sentence, trigger, event_type)

                event["arguments"] = pred_result["arguments"]

        save_json_data(data, os.path.join(self.prediction_path, f"pred_test1.{trigger_type}_trigger.json"))

        return data

    def predict_one_instance(self,
                             sentence: AnyStr,
                             trigger: Dict[AnyStr, Union[AnyStr, int]],
                             event_type: AnyStr
                             ) -> Dict[AnyStr, Union[AnyStr, Dict, int, float]]:
        trigger_text = sentence[trigger["start"]: trigger["end"]]
        sentence_marked = sentence[:trigger["start"]] \
                          + self.trigger_left_token \
                          + trigger_text \
                          + self.trigger_right_token \
                          + sentence[trigger["end"]:]

        prefix_len = len(trigger["text"] + self.tokenizer.sep_token)
        input_text = trigger["text"] + self.tokenizer.sep_token + sentence_marked

        input_text_encoded = self.tokenizer(
            input_text,
            max_length=self.config.max_position_embeddings,
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        input_text_encoded = input_text_encoded.to(self.device)

        offset_mapping = input_text_encoded.pop("offset_mapping")[0].cpu().numpy().tolist()

        pred_arguments = self.identify_arguments(input_text, input_text_encoded, offset_mapping)
        pred_argument_roles = self.classify_arguments(
            input_text_encoded=input_text_encoded,
            offset_mapping=offset_mapping,
            event_type=event_type,
            trigger=trigger,
            arguments=pred_arguments,
            prefix_len=prefix_len,
            sentence=sentence
        )

        pred_argument_roles = self.adjust_arguments_offset(
            arguments=pred_argument_roles,
            trigger=trigger,
            prefix_len=prefix_len
        )

        result = {
            "sentence": sentence,
            "trigger": trigger,
            "event_type": event_type,
            "arguments": pred_argument_roles
        }

        return result

    def identify_arguments(self,
                           input_text: AnyStr,
                           input_text_encoded: BatchEncoding,
                           offset_mapping: List[Tuple[int, int]]
                           ) -> List[Dict[AnyStr, Union[AnyStr, int]]]:
        start_mapping = {index: offset[0]
                         for index, offset in enumerate(offset_mapping)
                         if offset}
        end_mapping = {index: offset[-1]
                       for index, offset in enumerate(offset_mapping)
                       if offset}

        pred_label = self.model.predict_crf(**input_text_encoded)[0]
        pred_label = [self.args.index2tag[i] for i in pred_label]

        pred_arguments = []
        argument = {"text": None, "start": None, "end": None}

        for index, tag in enumerate(pred_label):
            if tag == "B":
                if argument["start"] is None:
                    argument["start"] = start_mapping[index]
                else:
                    argument["end"] = end_mapping[index - 1]
                    argument["text"] = input_text[argument["start"]: argument["end"]]
                    pred_arguments.append(argument)
                    argument = {"text": None, "start": None, "end": None}
            elif tag == "I":
                pass
            elif tag == "O" and argument["start"] is not None:
                argument["end"] = end_mapping[index - 1]
                argument["text"] = input_text[argument["start"]: argument["end"]]
                pred_arguments.append(argument)
                argument = {"text": None, "start": None, "end": None}

        return pred_arguments

    def classify_arguments(self,
                           input_text_encoded: BatchEncoding,
                           offset_mapping: List[Tuple[int, int]],
                           event_type: AnyStr,
                           trigger: Dict[AnyStr, Union[AnyStr, int]],
                           arguments: List[Dict[AnyStr, Union[AnyStr, int]]],
                           prefix_len: int,
                           sentence: AnyStr
                           ) -> List[Dict[AnyStr, Union[AnyStr, int, float]]]:
        if len(arguments) == 0:
            return []

        role_list = sorted(self.role_desc[event_type].keys())
        role_text_list = [self.role_desc[event_type][role] for role in role_list]

        start_mapping = {offset[0]: index
                         for index, offset in enumerate(offset_mapping)
                         if offset}
        end_mapping = {offset[-1]: index
                       for index, offset in enumerate(offset_mapping)
                       if offset}

        trigger_start = trigger["start"] + prefix_len + len(self.trigger_left_token)
        trigger_end = trigger["end"] + prefix_len + len(self.trigger_left_token)

        trigger_mask = torch.zeros_like(input_text_encoded["input_ids"]).bool()
        trigger_mask = trigger_mask.to(self.device)

        trigger_token_start = start_mapping[trigger_start]
        trigger_token_end = end_mapping[trigger_end]

        if trigger_token_end == trigger_token_end:
            trigger_mask[0][trigger_token_start] = True
        else:
            trigger_mask[0][trigger_token_start: trigger_token_end] = True

        role_encoded = self.tokenizer(
            role_text_list,
            truncation=True,
            max_length=self.args.max_role_length,
            padding="max_length",
            return_offsets_mapping=False,
            return_tensors="pt"
        )
        role_encoded = role_encoded.to(self.device)

        pred_argument_roles = []

        for argument in arguments:
            argument_mask = torch.zeros_like(input_text_encoded["input_ids"]).bool()
            argument_mask = argument_mask.to(self.device)
            argument_token_start = start_mapping[argument["start"]]
            argument_token_end = end_mapping[argument["end"]]

            if argument_token_start == argument_token_end:
                argument_mask[0][argument_token_start] = True
            else:
                argument_mask[0][argument_token_start: argument_token_end] = True

            probs = self.model.predict_kge(
                input_ids=input_text_encoded["input_ids"],
                token_type_ids=(input_text_encoded["token_type_ids"]
                                if "token_type_ids" in input_text_encoded.keys() else None),
                attention_mask=input_text_encoded["attention_mask"],
                trigger_mask=trigger_mask,
                argument_mask=argument_mask,
                role_input_ids=role_encoded["input_ids"],
                role_attention_mask=role_encoded["attention_mask"],
                role_token_type_ids=(role_encoded["token_type_ids"]
                                     if "token_type_ids" in role_encoded.keys() else None),
                has_pos_role=False
            )

            # ----------------------------------------------------------------------------
            # add_type_sim = False
            # type_gamma = 0.05
            # if add_type_sim:
            #     prompt_input_ids = self.prepare_prompt_input_ids(sentence, argument["text"])
            #     argument_vector = self.prompt_model.predict(prompt_input_ids)
            #     for i, role in enumerate(role_list):
            #         role_vector = self.role_type_vector[event_type][role]
            #         sim_score = torch.cosine_similarity(role_vector, argument_vector).item()
            #         # sim_score = torch.mul(role_vector, argument_vector).sum().item()
            #         probs[i] += type_gamma * sim_score
            # ----------------------------------------------------------------------------

            role_index = probs.argmax(dim=0)
            role_index = role_index.item()

            role_probs = probs.max(dim=0)
            role_probs = role_probs.values.item()

            argument = deepcopy(argument)
            argument["role"] = role_list[role_index][:-4]
            argument["probs"] = role_probs

            pred_argument_roles.append(argument)

        return pred_argument_roles

    @staticmethod
    def take_gold_trigger_as_input(data: List[Dict[AnyStr, Union[Dict, AnyStr, int]]]
                                   ) -> List[Dict[AnyStr, Union[Dict, AnyStr, int]]]:
        data = deepcopy(data)
        for instance in data:
            instance["pred_event_mentions"] = deepcopy(instance["event_mentions"])
            for event in instance["pred_event_mentions"]:
                event.pop("arguments")

        return data

    def adjust_arguments_offset(self,
                                arguments: List[Dict[AnyStr, Union[AnyStr, int, float]]],
                                trigger: Dict[AnyStr, Union[AnyStr, int]],
                                prefix_len: int
                                ) -> List[Dict[AnyStr, Union[AnyStr, int, float]]]:
        arguments = deepcopy(arguments)

        for argument in arguments:
            trigger_marked_start = trigger["start"] + prefix_len + len(self.trigger_left_token)
            trigger_marked_end = trigger["end"] + prefix_len + len(self.trigger_left_token)

            argument["start"] = self.adjust_index(argument["start"],
                                                  trigger_marked_start,
                                                  trigger_marked_end,
                                                  prefix_len)
            argument["end"] = self.adjust_index(argument["end"],
                                                trigger_marked_start,
                                                trigger_marked_end,
                                                prefix_len)

        return arguments

    def adjust_index(self,
                     index: int,
                     trigger_marked_start: int,
                     trigger_marked_end: int,
                     prefix_len: int
                     ) -> int:
        if index < trigger_marked_start:
            index -= prefix_len
        elif index <= trigger_marked_end:
            index -= prefix_len + len(self.trigger_left_token)
        else:
            index -= prefix_len \
                     + len(self.trigger_left_token) \
                     + len(self.trigger_right_token)

        return index
