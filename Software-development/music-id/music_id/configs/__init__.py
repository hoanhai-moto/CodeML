import os
import json
CONFIG_DEFAULT_FILE = 'config_default.json'
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(CONFIG_DIR)

def init(config_path=None):
    with open(os.path.join(CONFIG_DIR, config_path or CONFIG_DEFAULT_FILE)) as f:
        config = json.load(f)
    return config