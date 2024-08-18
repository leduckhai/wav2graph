import os
import re
import torch
from typing import Any

def clean_name(input_string) -> str:
    pattern = re.compile('[a-zA-Z]+')
    characters = ''.join(pattern.findall(input_string))
    return characters


def parameters_count(model:Any) -> int: 
    total_param = 0

    for param in model.parameters():
        total_param += param.numel()

    return total_param


def format_time(duration):
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))


def find_comet_api_key():
    if "COMET_API_KEY" in os.environ:
        return os.environ['COMET_API_KEY']

    return None


def generator(data:list[str], batch_size:int):
    total_samples = len(data)
    for i in range(0, total_samples, batch_size):
        if i + batch_size < total_samples:
            yield data[i:i + batch_size]
        else:
            yield data[i:]

def find_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"