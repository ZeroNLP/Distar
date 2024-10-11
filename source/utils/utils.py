import os
import json
import random
from typing import List, Dict, AnyStr

import numpy as np

import torch

from .logger import logger


def load_json_data(file_path: AnyStr) -> List[Dict]:
    data = []
    with open(file_path, "r") as file:
        for line in file:
            inst = json.loads(line)
            data.append(inst)
    logger.info(f"load {len(data)} instances from {os.path.abspath(file_path)}")

    return data


def save_json_data(data: List[Dict], file_path: AnyStr) -> None:
    with open(file_path, "w") as file:
        for inst in data:
            file.write(json.dumps(inst) + "\n")
            file.flush()
    logger.info(f"save {len(data)} instances to {os.path.abspath(file_path)}")


def load_split_types(file_path: AnyStr) -> Dict[AnyStr, List[AnyStr]]:
    with open(file_path, "r") as file:
        split_types = json.load(file)
    logger.info(f"load event type splits from {os.path.abspath(file_path)}")
    return split_types


def load_role_description(file_path: AnyStr,
                          punctuation: AnyStr = "."
                          ) -> Dict[AnyStr, Dict[AnyStr, AnyStr]]:
    with open(file_path, "r") as file:
        role_description = {}
        for line in file:
            line = line.strip()
            if line and line != "":
                if ":" not in line:
                    event_type = line
                    role_description[event_type] = {}
                else:
                    role, description = line.split(":")[0], line.split(":")[1]
                    role_description[event_type][role] = description + punctuation
    logger.info(f"load role description from {os.path.abspath(file_path)}")

    return role_description


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        logger.info(f"device name: {torch.cuda.get_device_name()}")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
