import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):

    def __init__(self,num_state_features:int,intermediate_dims:int,num_actions:int) -> None:

        super().__init__()

        self.fc1 = nn.Linear(num_state_features,intermediate_dims)
        self.fc2 = nn.Linear(intermediate_dims,intermediate_dims)
        self.fc3 = nn.Linear(intermediate_dims,num_actions)

    def forward(self,x:torch.Tensor) -> torch.Tensor:

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x   