import json
import os
from pathlib import Path
from easydict import EasyDict as edict


def get_config_regression(model_name, dataset_name, recovery_type, config_file=""):
    """
    Get the regression config of given dataset and model from config file.

    Parameters:
        config_file (str): Path to config file, if given an empty string, will use default config file.
        model_name (str): Name of model.
        dataset_name (str): Name of dataset.
        recovery_type (str): Type of recovery methods

    Returns:
        config (dict): config of the given dataset and model
    """
    if config_file == "":
        config_file = Path(__file__).parent / "configs" / "config.json"
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    model_common_args = config_all[model_name].get('commonParams', {})
    model_layer_args = config_all[model_name].get(recovery_type, {}) 
    dataset_args = config_all['datasetParams'].get(model_name, {})
    # use aligned feature if the model requires it, otherwise use unaligned feature

    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_layer_args)
    config['featPath'] = os.path.join(config_all['datasetParams']['dataset_root_dir'], config['featPath'])
    config['num_classes'] = config_all['datasetParams']['num_classes']
    
    config = edict(config)

    return config


