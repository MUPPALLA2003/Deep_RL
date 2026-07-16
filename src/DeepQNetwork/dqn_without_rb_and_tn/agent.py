import torch
import numpy as np
import torch.nn.functional as F
from .q_network import DQN_without_replaybuffer_and_targetnetwork

class Agent():

    def __init__(self,env,input_features:int,intermediate_dim:int,lr:float,gamma:float,eps:float,eps_dec:float,eps_min:float) -> None:

        num_actions = env.action_space.n
        self.input_features = input_features
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_Space = env.action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Q = DQN_without_replaybuffer_and_targetnetwork(input_features,num_actions,intermediate_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)
        
        
    def choose_action(self,observation):

        #assert observation.shape[-1] == self.input_features, "Last dimension should be the input_fetaures"

        observation = torch.tensor(observation,dtype=torch.float32,device = self.device)

        if np.random.random() > self.eps:

            with torch.no_grad():

                actions = self.Q(observation)

            action = torch.argmax(actions).item()

        else:

            action = self.action_Space.sample()

        return action

    def eps_decay(self):

        assert self.eps >= self.eps_min

        assert self.eps_dec >= 0

        self.eps = max(self.eps - self.eps_dec, self.eps_min)   

    def learn(self,state,action,reward,state_):

        states = torch.tensor(state,dtype = torch.float32,device = self.device)
        actions = torch.tensor(action,dtype = torch.long,device = self.device)
        rewards = torch.tensor(reward,dtype = torch.float32,device = self.device)
        states_ = torch.tensor(state_,dtype = torch.float32,device = self.device)
        q_pred = self.Q(states)[actions]

        with torch.no_grad():
            q_next = self.Q(states_).max()
            q_target = rewards + self.gamma * q_next

        loss = F.mse_loss(q_target,q_pred)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.eps_decay()
    
    