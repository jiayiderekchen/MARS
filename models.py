"""
MARS
Neural network architectures for trading agents and meta-controller.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import config

class Actor(nn.Module):
    """
    Actor network for determining trading actions.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        """
        Initialize the actor network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: Dimensions of hidden layers
        """
        super(Actor, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = config.ACTOR_HIDDEN_DIMS
        
        # Input layer
        layers = [nn.Linear(state_dim, hidden_dims[0]), nn.ReLU()]
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], action_dim))
        layers.append(nn.Tanh())  # Output in range [-1, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            Action tensor in range [-1, 1]
        """
        return self.network(state)


class Critic(nn.Module):
    """
    Critic network for estimating state-action values.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        """
        Initialize the critic network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: Dimensions of hidden layers
        """
        super(Critic, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = config.CRITIC_HIDDEN_DIMS
        
        # Input layer (state + action)
        self.input_layer = nn.Linear(state_dim + action_dim, hidden_dims[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Q-value tensor
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.input_layer(x))
        
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        return self.output_layer(x)


class SafetyCritic(nn.Module):
    """
    Safety critic network for estimating risk of state-action pairs.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        """
        Initialize the safety critic network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: Dimensions of hidden layers
        """
        super(SafetyCritic, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = config.SAFETY_CRITIC_HIDDEN_DIMS
        
        # Input layer (state + action)
        self.input_layer = nn.Linear(state_dim + action_dim, hidden_dims[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Risk value tensor in range [0, 1]
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.input_layer(x))
        
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        return torch.sigmoid(self.output_layer(x))


class MetaController(nn.Module):
    """
    Meta-controller network for determining agent weights.
    """
    def __init__(self, state_dim: int, num_agents: int):
        """
        Initialize the meta-controller network.
        
        Args:
            state_dim: Dimension of the state space
            num_agents: Number of agents to weight
        """
        super(MetaController, self).__init__()
        
        # Network architecture
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_agents)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            Agent weights tensor (will be softmaxed externally)
        """
        return self.network(state) 
