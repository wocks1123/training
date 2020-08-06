import json


def load_config(config_path):
    with open(config_path, "r") as fp:
        config = json.load(fp)

    return config
