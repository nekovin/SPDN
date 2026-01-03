import yaml
import torch.nn as nn


def parse_config(config, override_args=None):
    parsed_config = {}

    for section, values in config.items():
        parsed_config[section] = {}
        for key, value in values.items():
            parsed_config[section][key] = _convert_value(key, value)

    if override_args:
        for section, values in override_args.items():
            if section not in parsed_config:
                parsed_config[section] = {}

            for key, value in values.items():
                parsed_config[section][key] = _convert_value(key, value)

    return parsed_config


def _convert_value(key, value):
    """Helper function to convert configuration values to appropriate types."""
    if key == "learning_rate" and isinstance(value, str):
        return float(value)

    elif key == "criterion" and isinstance(value, str):
        if value == "nn.MSELoss()":
            return nn.MSELoss()
        elif value == "nn.L1Loss()":
            return nn.L1Loss()
        else:
            return value

    elif key == "optimizer" and isinstance(value, str):
        return value

    else:
        return value


def get_config(config_path, override_args=None):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return parse_config(config, override_args)
