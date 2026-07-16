import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN_without_replaybuffer_and_targetnetwork(nn.Module):

    def __init__(self,input_features:int,num_actions:int,intermediate_dim:int) -> None:

        super(DQN_without_replaybuffer_and_targetnetwork,self).__init__()

        self.fc1 = nn.Linear(input_features,intermediate_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(intermediate_dim,num_actions)

    def forward(self,state:torch.Tensor) -> torch.Tensor:

        #assert state.shape[-1] == self.input_features

        layer1 = self.relu(self.fc1(state))
        actions = self.fc2(layer1)

        return actions