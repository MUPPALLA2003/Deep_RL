import torch
import torch.nn as nn
import torch.optim as optim
from .network import Network
from .replay_buffer import ReplayBuffer
import numpy as np

class Agent:

    def __init__(self,max_memories:int,discount_factor:float,learning_rate:float,num_state_features:int,num_actions:int,intermediate_dims:int,epsilon:float,epsilon_decay:float,min_epsilon:float,device:str) -> None:

        self.max_memories = max_memories
        self.discount_factor = discount_factor 
        self.learning_rate = learning_rate
        self.num_state_features = num_state_features
        self.num_actions = num_actions
        self.action_space = np.arange(self.num_actions)
        self.intermediate_dims = intermediate_dims
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon 
        self.device = device
        self.DQN = Network(num_state_features,intermediate_dims,num_actions).to(device)
        self.optimizer = optim.Adam(self.DQN.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(num_state_features,max_memories,device)

    def select_action(self,state):

        if not isinstance(state,torch.Tensor):

            state = torch.tensor(state,device=self.device)

        if state.dim() == 1:

            state = state.unsqueeze(0)

        assert state.shape[-1] == self.num_state_features, f"Passing {state.shape[-1]} features but expected {self.num_state_features}"
        
        if np.random.rand() < self.epsilon:

            action = np.random.choice(self.action_space)

        else:

            self.DQN.eval()

            with torch.no_grad():

                Q_sa = self.DQN(state)

            action = torch.argmax(Q_sa).item()

            self.DQN.train()

        return action

    def update_epsilon(self):
        
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def inference(self,state):

        assert self.DQN.device == self.device

        self.DQN = self.DQN.to(self.device)

        self.DQN.eval()
        
        with torch.no_grad():

            Q_s_a = self.DQN(state.to(self.device))
            
        return torch.argmax(Q_s_a).item()
        
    def train_step(self,batch_size):

        batch = self.replay_buffer.access_memory(batch_size)

        if batch is None:

            return None

        q_estimate = self.DQN(batch["states"])
        q_estimate = torch.gather(q_estimate, index=batch["actions"].unsqueeze(-1), dim=-1).squeeze(-1) 

        with torch.no_grad():
        
            self.DQN.eval()

            q_next_estimate = self.DQN(batch["next_states"])

            self.DQN.train()

        max_q_next_estimate = torch.max(q_next_estimate, dim=-1).values
        td_target = batch["rewards"] + self.discount_factor * max_q_next_estimate * (~batch["terminals"])
        loss = self.loss_fn(td_target, q_estimate)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.update_epsilon()