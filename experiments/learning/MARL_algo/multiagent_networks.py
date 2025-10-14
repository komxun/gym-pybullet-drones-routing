import torch
from torch import nn
from typing import List
from gym import spaces
class MultiAgentFCNetwork_SharedParameters(nn.Module):

    def __init__(
            self,
            in_sizes: List[int],
            out_sizes: List[int]
        ):
        super().__init__()
        
        activ = nn.ReLU                   # Use ReLU activation function:
        hidden_dims = (64, 64)            # Use 2 hidden layers of 64 units each:
        n_agents = len(in_sizes)          # number of agent is the length of input and ouput vector
        assert n_agents == len(out_sizes) # We will create 'n_agents' (independent) networks

        # For each agent
        for in_size, out_size in zip(in_sizes, out_sizes):
            network = [
                nn.Linear(in_size, hidden_dims[0]),
                activ(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                activ(),
                nn.Linear(hidden_dims[1], out_size),
            ]
            self.networks.append(nn.Sequential(*network))
    
    def forward(self, inputs: List[torch.Tensor]):
        # The networks can run in parallel
        futures = [
            torch.jit.fork(model, inputs[i]) for i, model in enumerate(self.networks)
        ]
        results = [torch.jit.wait(fut) for fut in futures]
        return results
    
#########################################################################

class MultiAgentFCNetwork_SharedParameters(nn.Module):
    def __init__(
            self,
            in_sizes: List[int],
            out_sizes: List[int]
        ):
        super().__init__()
        
        activ = nn.ReLU                   # Use ReLU activation function:
        hidden_dims = (64, 64)            # Use 2 hidden layers of 64 units each:
        n_agents = len(in_sizes)          # number of agent is the length of input and ouput vector
        assert n_agents == len(out_sizes) # We will create 'n_agents' (independent) networks

        # Only ONE SHARED NETWORK 
        # This assumes that input and ouput size of the networks is identical for all agents
        
        network = [
            nn.Linear(in_sizes, hidden_dims[0]),
            activ(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            activ(),
            nn.Linear(hidden_dims[1], out_sizes),
        ]
        self.networks = nn.Sequential(*network)
        
    def forward(self, inputs: List[torch.Tensor]):
        futures = [
            torch.jit.fork(self.network, inp) for inp in inputs
        ]
        results = [torch.jit.wait(fut) for fut in futures]
        return results

        
