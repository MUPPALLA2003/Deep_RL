import torch
import numpy as np
from typing import Dict

class ReplayBuffer:

    def __init__(self,max_memories:int,num_state_features:int,device:str) -> None:

        self.max_memories = max_memories
        self.device = device
        self.memories_counter = 0
        self.state_mem = torch.zeros(max_memories,num_state_features,dtype = torch.float32)
        self.next_state_mem = torch.zeros(max_memories,num_state_features,dtype = torch.float32)
        self.action_mem = torch.zeros(max_memories,dtype = torch.long)
        self.reward_mem = torch.zeros(max_memories,dtype = torch.float32)
        self.terminal_mem = torch.zeros(max_memories,dtype = torch.bool)

    def update_memory(self,state,next_state,action,reward,terminal):

        total_memories = max(self.total_memories,self.memories_counter)
        idx = total_memories % self.max_memories

        self.state_mem[idx] = torch.tensor(state,dtype = self.state_mem.dtype)
        self.next_state_mem[idx] = torch.tensor(next_state,dtype = self.next_state_mem.dtype)
        self.action_mem[idx] = torch.tensor(action,dtype = self.action_mem.dtype)
        self.reward_mem[idx] = torch.tensor(reward,dtype = self.reward_mem.dtype)
        self.terminal_mem[idx] = torch.tensor(terminal,dtype = self.terminal_mem.dtype)

        self.memories_counter += 1

    def access_memories(self,batch:int) -> Dict[str,torch.Tensor]:

        assert batch < self.max_memories, "Batch size is more than the memories we can hold at once"

        self.total_memories = min(self.memories_counter,self.max_memories)

        if self.total_memories < batch:

            return None

        rand_idx = np.random.choice(np.arange(self.total_memories),size = batch,replace=False)
        rand_idx = torch.tensor(rand_idx,dtype = torch.long)

        batch = {
            "states" : self.state_mem[rand_idx].to(self.device),
            "next_states" : self.next_state_mem[rand_idx].to(self.device),
            "actions" : self.action_mem[rand_idx].to(self.device),
            "rewards" : self.reward_mem[rand_idx].to(self.device),
            "terminals" : self.terminal_mem[rand_idx].to(self.device),
        }

        return batch




        

             