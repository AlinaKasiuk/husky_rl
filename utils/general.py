import yaml
import random

import numpy as np

from agents.agent import DeepQAgent


def load_config_data(config_file: str) -> dict:
    with open(config_file) as fd:
        return yaml.safe_load(fd)


def select_action(agent: DeepQAgent, observation: np.array, action_eps: float, action: int):
    """
    Select the action according to the agent model.
    With probability action_eps, keep the same action.
    """
    if random.random() < action_eps:
        return action
    return agent.predict_action(observation)


def get_str_device(param_device: str):
    if param_device == 'cpu' or param_device == 'cuda':
        return param_device
    assert param_device.isdigit(), "Unrecognized device!!"

    return 'cuda:{0}'.format(param_device)
