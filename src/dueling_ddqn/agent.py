import torch
import torch.nn as nn
import numpy as np
from .network import Network
from .replay_buffer import ReplayBuffer
import torch.optim as optim

class Agent:

    def __init__(self,max_memories:int,discount_factor:float,learning_rate:float,num_state_features:int,intermediate_dims:int,num_actions:int,epsilon:float,epsilon_decay:float,min_epsilon:float,device:str) -> None:

        self.max_memories = max_memories
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.num_state_features = num_state_features
        self.intermediate_dims = intermediate_dims
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.device = device
        self.action_space = np.arange(self.num_actions)
        self.dqn = Network(num_state_features,intermediate_dims,num_actions).to(device)
        self.target_dqn = Network(num_state_features,intermediate_dims,num_actions).to(device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()
        self.optimizer = optim.Adam(self.dqn.parameters(),lr = learning_rate)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(max_memories,num_state_features,device)

    def select_action(self,state):

        if not isinstance(state,torch.Tensor):

            state = torch.tensor(state,device = self.device)

        if state.dim() == 1:

            state = state.unsqueeze(0)

        assert state.shape[-1] == self.num_state_features, f"Passing {state.shape[-1]} features but expected {self.num_state_features}"    

        if np.random.rand() < self.epsilon:

            action = np.random.choice(self.action_space)

        else:

            self.dqn.eval()

            with torch.no_grad():

                Q_sa = self.dqn(state)

            action = torch.argmax(Q_sa).item()
            self.dqn.train() 

        return action

    def update_epsilon(self):

        self.epsilon = max(self.min_epsilon,self.epsilon * self.epsilon_decay)

    def inference(self,state):

        with torch.no_grad():

            q_sa = self.dqn(state.to(self.device))    

        return torch.argmax(q_sa).item()

    def update_target_network(self):

        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def train_step(self,batch_size:int):

        batch = self.replay_buffer.access_memories(batch_size)

        if batch is None:

            return None

        self.dqn.train() 

        q_estimate = self.dqn(batch["states"])
        q_estimate = torch.gather(q_estimate,index = batch["actions"].unsqueeze(-1),dim = 1).squeeze(-1)

        with torch.no_grad():

            q_next_estimate = self.dqn(batch["next_states"])
            next_actions = torch.argmax(q_next_estimate,dim=-1,keepdim=True)
            next_q_target_values = self.target_dqn(batch['next_states'])
            max_q_next_estimate = next_q_target_values.gather(dim =-1,index =next_actions).squeeze(-1)

        td_target = batch["rewards"] + self.discount_factor * max_q_next_estimate * (~batch["terminals"])
        loss = self.loss_fn(td_target,q_estimate)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(),max_norm = 1.0)
        self.optimizer.step()
        self.update_epsilon()