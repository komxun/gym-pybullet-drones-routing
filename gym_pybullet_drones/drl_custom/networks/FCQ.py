from gym_pybullet_drones.drl_custom.drl_imports import torch, nn, F

class FCQ(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_dims=(32,32), 
                 activation_fc=F.relu):
        super(FCQ, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)
        
    # def _format(self, state):
    #     x = state
    #     print(f"state: {x}")
    #     if not isinstance(x, torch.Tensor):
    #         x = torch.tensor(x, 
    #                          device=self.device, 
    #                          dtype=torch.float32)
    #         x = x.unsqueeze(0)
    #     return x

    def _format(self, state):
        # Handle tuple case, where state might be (array, dict)
        if isinstance(state, tuple):
            state = state[0]  # Extract the first part, which is the state array

        # Check if it's a tensor, if not convert it
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, 
                                device=self.device, 
                                dtype=torch.float32)
            # Ensure it has the correct shape (batch_size, input_dim)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)  # Add batch dimension if it's a single state
        
        return state

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        return x
    
    def numpy_float_to_device(self, variable):
        variable = torch.from_numpy(variable).float().to(self.device)
        return variable
    
    def load(self, experiences):
        states, actions, new_states, rewards, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, new_states, rewards, is_terminals