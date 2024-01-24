import torch
from torch.optim import AdamW
from torch.nn import SmoothL1Loss

import copy
import numpy as np

from agents.anns.nets import get_torch_model, load_model


class DeepQAgent():

    def __init__(self, num_classes: int, \
        model_name: str, device, weight_path: str='') -> None:
        """
        The agent learn from the environment using a CNN.
        
        @param num_classes: the number of classes (possible actions).
        @param model_name: the name of the model to use. It must be a valid model implemented in the torch library.
        @param weight_path: if provided a valid path, then load this model instead of using the torch model.
        """
        try:
            model = load_model(weight_path)
            model_name = "weight_model"
        except:
            model = get_torch_model(model_name, num_classes)
        
        assert model is not None, "Invalid model definition."

        self.model_name = model_name
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.device = device

        self.model.to(device)
        self.target_model.to(device)

        self.criterion = SmoothL1Loss()
        self.optimizer = AdamW(self.model.parameters(), lr=0.001)

        self.gamma = 0.8
        self.train_iterations = 0

    def train(self, data: np.array) -> None:
        self.model.train()
        self.target_model.eval()

        # prepare data for train
        current_obs = np.stack(data[:, 0])
        next_obs = np.stack(data[:, 1])
        actions = data[:, 2].astype('int')
        rewards = data[:, 3].astype('float')
        dones = data[:, 4].astype('bool')

        model_input = torch.from_numpy(current_obs).to(self.device).float()
        target_input = torch.from_numpy(next_obs).to(self.device).float()

        # do one train step
        self.optimizer.zero_grad()

        # predict the q values for the current state
        y_hat = self.model(model_input)
        # select the performed actions
        y_hat = y_hat[np.arange(len(data)), actions]
        # predict the q values for the next state
        with torch.no_grad():
            y_target = self.target_model(target_input)
        # select the maximum values
        y_target = y_target.max(dim=1)[0]
        y_target[dones] = 0.0
        # the q values for the current state follow the Bellman equation
        target_q_value = torch.from_numpy(rewards).to(self.device) + self.gamma * y_target

        loss = self.criterion(target_q_value, y_hat)
        loss.backward()
        self.optimizer.step()

        self.train_iterations += 1
        #

    def replace_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.to(self.device)

    @torch.no_grad()
    def predict_action(self, observation: np.array) -> int:
        model_input = torch.from_numpy(observation).to(self.device).float().unsqueeze(0)

        y_hat = self.model(model_input).squeeze(0)

        return y_hat.argmax().item()
