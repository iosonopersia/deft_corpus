import os
import random

import numpy as np
import torch
from munch import Munch

config: Munch = None
config_path: str = './definitions_classifier/config.yaml'

def get_config() -> Munch:
    global config
    if config is None:
        if not os.path.exists(config_path):
            raise FileNotFoundError('Config file not found')

        with open(config_path, 'rt', encoding='utf-8') as f:
            config = Munch.fromYAML(f.read())
    return config


def seed_everything(seed: int=42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
