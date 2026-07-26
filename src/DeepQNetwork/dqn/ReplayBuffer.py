import torch
import numpy as np
from typing import Dict

class ReplayBuffer:

    def __init__(self,num_state_features:int,max_memories:int,device:str) -> None:

        self.num_state_features = num_state_features
        self.max_memories = max_memories
        self.device = device
        self.current_memories_counter = 0
        self.state_mem = torch.zeros(max_memories,num_state_features,dtype = torch.float32)
        self.next_state_mem = torch.zeros(max_memories,num_state_features,dtype = torch.float32)
        self.reward_mem = torch.zeros(max_memories,dtype = torch.float32)
        self.action_mem = torch.zeros(max_memories,dtype = torch.long)
        self.terminal_mem = torch.zeros(max_memories,dtype = torch.bool)

    def update_memory(self,state,next_state,action,reward,terminal) -> None:

        idx = self.current_memories_counter % self.max_memories

        self.state_mem[idx] = torch.tensor(state,dtype = self.state_mem.dtype)
        self.next_state_mem[idx] = torch.tensor(next_state,dtype = self.next_state_mem.dtype)
        self.action_mem[idx] = torch.tensor(action,dtype = self.action_mem.dtype)
        self.reward_mem[idx] = torch.tensor(reward,dtype = self.reward_mem.dtype)
        self.terminal_mem[idx] = torch.tensor(terminal,dtype = self.terminal.dtype)

        self.current_memories_counter += 1

    def access_memory(self,batch_size:int) -> Dict[str,torch.Tensor]:

        assert batch_size < self.max_memories

        total_memories = min(self.current_memories_counter,self.max_memories)

        if total_memories < batch_size:

            return None

        rand_sample_idx = np.random.choice(np.arange(total_memories),size = batch_size,replace = False)
        rand_sample_idx = torch.tensor(rand_sample_idx,dtype = torch.long)

        batch = {
            "states":self.state_mem[rand_sample_idx].to(self.device),
            "next_states":self.next_state_mem[rand_sample_idx].to(self.device),
            "actions":self.action_mem[rand_sample_idx].to(self.device),
            "rewards":self.reward_mem[rand_sample_idx].to(self.device),
            "terminals":self.terminal_mem[rand_sample_idx].to(self.device),
        }

        return batch




            

