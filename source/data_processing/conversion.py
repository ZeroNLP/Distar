from copy import deepcopy
from typing import AnyStr, List, Tuple, Dict, Any

from tqdm.auto import tqdm


def get_span_map(sentence: AnyStr,
                 tokens: List[AnyStr]
                 ) -> Tuple[Dict, Dict]:
    start_map = {}
    end_map = {}

    curr_index = 0
    for index, token in enumerate(tokens):
        start_index = sentence.find(token, curr_index)
        end_index = start_index + len(token)
        curr_index = end_index

        start_map[index] = start_index
        end_map[index] = end_index

    return start_map, end_map


def process_event_span(event_mention: Dict[AnyStr, Any],
                       sentence: AnyStr,
                       start_map: Dict[int, int],
                       end_map: Dict[int, int]
                       ) -> Dict[AnyStr, Any]:
    event_mention = deepcopy(event_mention)
    trigger_item = event_mention["trigger"]
    trigger_start = trigger_item["start"]
    trigger_end = trigger_item["end"]

    # if trigger_item["text"] != \
    #         sentence[start_map[trigger_start]: end_map[trigger_end - 1]]:
    #     raise ValueError(f"Failed to update offsets. "
    #                      f"trigger: {trigger_item['text']}, offset: "
    #                      f"{sentence[start_map[trigger_start]: end_map[trigger_end - 1]]}")

    event_mention["trigger"]["start"] = start_map[trigger_start]
    event_mention["trigger"]["end"] = end_map[trigger_end - 1]

    for arg_item in event_mention["arguments"]:
        arg_start = arg_item["start"]
        arg_end = arg_item["end"]

        if arg_item["text"] != \
                sentence[start_map[arg_start]: end_map[arg_end - 1]]:
            raise ValueError(f"Failed to update offsets. "
                             f"argument: {arg_item['text']}, offset: "
                             f"{sentence[start_map[arg_start]: end_map[arg_end - 1]]}")

        arg_item["start"] = start_map[arg_start]
        arg_item["end"] = end_map[arg_end - 1]

    return event_mention


def split_by_event_type(data: List[Dict[AnyStr, Any]],
                        type_split: Dict[AnyStr, List[AnyStr]],
                        is_char_offset: bool = True
                        ) -> Dict[AnyStr, List[Dict[AnyStr, Any]]]:
    data_dict = {"seen": [], "unseen": []}

    for inst in tqdm(data, desc="splitting by type"):
        sentence = inst["sentence"]
        tokens = inst["tokens"]

        start_map = None
        end_map = None
        if not is_char_offset:
            start_map, end_map = get_span_map(sentence, tokens)

        for event in inst["event_mentions"]:
            inst_copy = deepcopy(inst)
            if not is_char_offset:
                event = process_event_span(event, sentence, start_map, end_map)
            inst_copy["event_mentions"] = [event]

            if event["event_type"] in type_split["seen_types"]:
                data_dict["seen"].append(inst_copy)
            else:
                data_dict["unseen"].append(inst_copy)

    print(f"seen data: {len(data_dict['seen'])} samples")
    print(f"unseen data: {len(data_dict['unseen'])} samples")

    return data_dict


def find_data_index(data: List[Dict[AnyStr, Any]],
                    inst: Dict[AnyStr, Any]
                    ) -> int:
    for idx, dt in enumerate(data):
        if dt["sentence"] == inst["sentence"]:
            return idx
    return -1


def merge_unseen_event(data: List[Dict[AnyStr, Any]]) -> List[Dict[AnyStr, Any]]:
    data_merged = []

    for inst in tqdm(data, desc="merging unseen data"):
        inst = deepcopy(inst)
        index = find_data_index(data_merged, inst)
        if index == -1:
            data_merged.append(inst)
        else:
            data_merged[index]["event_mentions"].extend(inst["event_mentions"])

    return data_merged


def convert_seen_data(data: List[Dict[AnyStr, Any]]) -> List[Dict[AnyStr, Any]]:
    data_processed = []

    for inst in tqdm(data, "converting seen data"):
        sentence = inst["sentence"]

        for event in inst["event_mentions"]:
            event_inst = {
                "sentence": sentence,
                "trigger": event["trigger"],
                "event_type": event["event_type"],
                "arguments": event["arguments"]
            }

            for argument in event["arguments"]:
                role = argument["role"]
                inst_processed = deepcopy(event_inst)
                inst_processed["triplet"] = {
                    "argument": argument,
                    "trigger": event["trigger"],
                    "role": role,
                    "label": 1
                }
                data_processed.append(inst_processed)

    return data_processed

