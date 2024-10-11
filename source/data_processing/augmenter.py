import json
import string
import random
from copy import deepcopy
from typing import Optional, Dict, Union, AnyStr, List, Tuple

from tqdm.auto import tqdm
from nltk import word_tokenize, pos_tag

from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration


class ArgumentAugmenter(object):
    def __init__(self,
                 tokenizer: T5Tokenizer,
                 config: T5Config,
                 model: T5ForConditionalGeneration,
                 num_beam: Optional[int] = 200,
                 num_return: Optional[int] = 15,
                 max_argument_length: Optional[int] = 10
                 ) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.model = model
        self.num_beam = num_beam
        self.num_return = num_return
        self.max_argument_length = max_argument_length
        self.device = self.model.device
        self.model.eval()

        self.mask_token = tokenizer.additional_special_tokens[0]
        self.end_token = tokenizer.additional_special_tokens[1]

    def generate_augmented_data(self,
                                data: List[Dict[AnyStr, Union[Dict, AnyStr]]]
                                ) -> List[Dict[AnyStr, Union[Dict, AnyStr]]]:
        augmented_data = []
        for instance in tqdm(data, desc="augmenting data"):
            augmented_instance, aug_score, replace_list = self.generate_augmented_instance(instance)
            if instance != augmented_instance:
                augmented_instance["aug_score"] = aug_score
                augmented_instance["replace_list"] = replace_list
                augmented_data.append(augmented_instance)

        augmented_data = sorted(augmented_data, key=lambda x: x["aug_score"], reverse=True)

        sample_num = 50
        for a in augmented_data[:50]:
            replace_list = a.pop("replace_list")
            print("-" * 100)
            print()
            print(a["sentence"])
            print()
            if sample_num <= 0:
                break
            if len(replace_list) <= sample_num:
                print("\n".join(replace_list))
                sample_num -= len(replace_list)
            else:
                print("\n".join(replace_list[:sample_num - len(replace_list)]))
                sample_num = 0
            print()
            print()

        return augmented_data

    def generate_augmented_instance(self,
                                    instance: Dict[AnyStr, Union[Dict, AnyStr]]
                                    ) -> Tuple[Dict[AnyStr, Union[Dict, AnyStr]], float, List]:
        augmented_instance = deepcopy(instance)
        sentence = augmented_instance["sentence"]
        augmented_sentence = deepcopy(sentence)

        # ----------------------------------
        replace_list = []

        aug_score = 0.0
        aug_count = 0

        for event in augmented_instance["event_mentions"]:
            trigger = event["trigger"]
            arguments = event["arguments"]

            arguments = sorted(arguments, key=lambda x: x["start"])
            arguments_prev = deepcopy(arguments)

            for i in range(len(arguments)):
                argument = arguments_prev[i]
                if not (trigger["start"] >= argument["end"]
                        or trigger["end"] <= argument["start"]):
                    continue

                augmented_argument, seq_score = self.generate_argument(augmented_sentence, arguments_prev[i])
                if augmented_argument is None:
                    continue

                replace_list.append(f"{argument['text']} \t-> {augmented_argument}  \t {argument['role']}")

                aug_score += seq_score
                aug_count += 1
                move_index = len(augmented_argument) - len(argument["text"])

                if trigger["start"] >= argument["end"]:
                    trigger["start"] = trigger["start"] + move_index
                    trigger["end"] = trigger["end"] + move_index

                augmented_sentence = augmented_sentence[:argument["start"]] \
                                     + augmented_argument \
                                     + augmented_sentence[argument["end"]:]
                arguments_prev[i]["text"] = augmented_argument
                arguments_prev[i]["end"] = argument["start"] + len(augmented_argument)

                for j in range(i + 1, len(arguments_prev)):
                    follow_argument = arguments_prev[j]
                    follow_argument["start"] = follow_argument["start"] + move_index
                    follow_argument["end"] = follow_argument["end"] + move_index

                augmented_instance["sentence"] = augmented_sentence
            event["arguments"] = arguments_prev

        return augmented_instance, aug_score / (aug_count + 1e-6), replace_list

    def generate_argument(self,
                          sentence: AnyStr,
                          argument: Dict[AnyStr, Union[AnyStr, int]]
                          ) -> Tuple[AnyStr, float]:
        sentence_masked = sentence[:argument["start"]] + self.mask_token + sentence[argument["end"]:]

        text_encoded = self.tokenizer.encode_plus(
            sentence_masked,
            add_special_tokens=True,
            return_tensors="pt"
        )
        input_ids = text_encoded["input_ids"].to(self.device)

        outputs = self.model.generate(input_ids=input_ids,
                                      num_beams=self.num_beam,
                                      max_length=self.max_argument_length,
                                      num_return_sequences=self.num_return,
                                      return_dict_in_generate=True,
                                      # use_cache=True,
                                      output_scores=True)

        pred_arguments = {}
        for index, output in enumerate(outputs.sequences):
            output_text = self.tokenizer.decode(
                output[2:],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )
            if self.end_token in output_text:
                end_token_index = output_text.index(self.end_token)
                pred_argument = output_text[:end_token_index].strip()

                argument_pos = pos_tag(word_tokenize(pred_argument))
                if (pred_argument != argument["text"]) \
                        and (pred_argument not in string.punctuation) \
                        and (pred_argument[0] not in string.punctuation) \
                        and (pred_argument[-1] not in string.punctuation) \
                        and (argument_pos[-1][-1].startswith(("NN", "PRP", "CD"))
                             or argument_pos[-1][0] in ["there", "here"]):

                    score = outputs.sequences_scores[index].item()
                    if pred_argument in pred_arguments.keys():
                        pred_arguments[pred_argument] = max(pred_arguments[pred_argument], score)
                    else:
                        pred_arguments[pred_argument] = score

        pred_argument_list = sorted(pred_arguments.items(), key=lambda x: x[-1], reverse=True)

        augmented_argument = None
        score = float("-inf")
        if len(pred_argument_list) > 0:
            # select_index = random.randint(0, len(pred_argument_list) - 1)
            # augmented_argument, score = pred_argument_list[select_index]
            augmented_argument, score = pred_argument_list[0]

        # print(sentence)
        # print(sentence_masked)
        # print(pred_argument_list)
        # for sent in pred_argument_list:
        #     print(pos_tag(word_tokenize(sent[0])))

        return augmented_argument, score

#
# class ArgumentAugmenter(object):
#     def __init__(self,
#                  tokenizer: T5Tokenizer,
#                  config: T5Config,
#                  model: T5ForConditionalGeneration,
#                  num_beam: Optional[int] = 200,
#                  num_return: Optional[int] = 15,
#                  max_argument_length: Optional[int] = 10
#                  ) -> None:
#         self.tokenizer = tokenizer
#         self.config = config
#         self.model = model
#         self.num_beam = num_beam
#         self.num_return = num_return
#         self.max_argument_length = max_argument_length
#         self.device = self.model.device
#
#         self.mask_token = tokenizer.additional_special_tokens[0]
#         self.end_token = tokenizer.additional_special_tokens[1]
#
#     def generate_augmented_data(self,
#                                 data: List[Dict[AnyStr, Union[Dict, AnyStr]]]
#                                 ) -> List[Dict[AnyStr, Union[Dict, AnyStr]]]:
#         augmented_data = []
#         for instance in tqdm(data, desc="augmenting data"):
#             augmented_instance = self.generate_augmented_instance(instance)
#             if instance != augmented_instance:
#                 augmented_data.append(augmented_instance)
#
#         return augmented_data
#
#     def generate_augmented_instance(self,
#                                     instance: Dict[AnyStr, Union[Dict, AnyStr]]
#                                     ) -> Dict[AnyStr, Union[Dict, AnyStr]]:
#         augmented_instance = deepcopy(instance)
#         sentence = augmented_instance["sentence"]
#         augmented_sentence = deepcopy(sentence)
#
#         for event in augmented_instance["event_mentions"]:
#             trigger = event["trigger"]
#             arguments = event["arguments"]
#
#             arguments = sorted(arguments, key=lambda x: x["start"])
#             arguments_prev = deepcopy(arguments)
#
#             for i in range(len(arguments)):
#                 argument = arguments[i]
#                 if not (trigger["start"] >= argument["end"]
#                         or trigger["end"] <= argument["start"]):
#                     continue
#
#                 augmented_argument = self.generate_argument(sentence, arguments_prev[i])
#                 if augmented_argument is None:
#                     continue
#
#                 move_index = len(augmented_argument) - len(argument["text"])
#
#                 if trigger["start"] >= argument["end"]:
#                     trigger["start"] = trigger["start"] + move_index
#                     trigger["end"] = trigger["end"] + move_index
#
#                 augmented_sentence = augmented_sentence[:argument["start"]] \
#                                      + augmented_argument \
#                                      + augmented_sentence[argument["end"]:]
#                 argument["text"] = augmented_argument
#                 argument["end"] = argument["start"] + len(augmented_argument)
#
#                 for j in range(i + 1, len(arguments)):
#                     follow_argument = arguments[j]
#                     follow_argument["start"] = follow_argument["start"] + move_index
#                     follow_argument["end"] = follow_argument["end"] + move_index
#
#         augmented_instance["sentence"] = augmented_sentence
#
#         return augmented_instance
#
#     def generate_argument(self,
#                           sentence: AnyStr,
#                           argument: Dict[AnyStr, Union[AnyStr, int]]
#                           ) -> AnyStr:
#         sentence_masked = sentence[:argument["start"]] + self.mask_token + sentence[argument["end"]:]
#
#         text_encoded = self.tokenizer.encode_plus(
#             sentence_masked,
#             add_special_tokens=True,
#             return_tensors="pt"
#         )
#         input_ids = text_encoded["input_ids"].to(self.device)
#
#         outputs = self.model.generate(input_ids=input_ids,
#                                       num_beams=self.num_beam,
#                                       max_length=self.max_argument_length,
#                                       num_return_sequences=self.num_return,
#                                       return_dict_in_generate=True,
#                                       output_scores=True)
#
#         pred_arguments = []
#         for output in outputs.sequences:
#             output_text = self.tokenizer.decode(
#                 output[2:],
#                 skip_special_tokens=False,
#                 clean_up_tokenization_spaces=False
#             )
#             if self.end_token in output_text:
#                 end_token_index = output_text.index(self.end_token)
#                 pred_argument = output_text[:end_token_index].strip()
#
#                 if (pred_argument != argument["text"]) and (pred_argument not in string.punctuation) \
#                         and (pred_argument[0] not in string.punctuation) \
#                         and (pred_argument[-1] not in string.punctuation):
#                     pred_arguments.append(pred_argument)
#
#         # pred_arguments = sorted(list(set(pred_arguments)))
#         augmented_argument = pred_arguments[0] if len(pred_arguments) > 0 else None
#
#         print(sentence)
#         print(sentence_masked)
#         print(pred_arguments)
#         for sent in pred_arguments:
#             print(pos_tag(word_tokenize(sent)))
#
#         return augmented_argument
