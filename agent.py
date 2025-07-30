"""
MARS: Multi-Agent Reinforcement Strategy
Safety-critic agent implementation for risk-aware trading decisions.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import random
from collections import deque
import config
from models import Actor, Critic, SafetyCritic
import utils

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    """
    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool, unsafe: bool):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            unsafe: Whether the action was unsafe
        """
        self.buffer.append((state, action, reward, next_state, done, unsafe))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Batch of transitions
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones, unsafes = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            np.array(unsafes, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        """
        Get the current size of the buffer.
        
        Returns:
            Number of transitions in the buffer
        """
        return len(self.buffer)


class SafetyCriticAgent:
    """
    Agent that uses a safety critic to avoid unsafe actions.
    """
    def __init__(self, state_dim: int, action_dim: int, agent_id: int, 
                 safety_threshold: float, safety_weight: float, device: str = "cpu"):
        """
        Initialize the safety critic agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            agent_id: Unique identifier for the agent
            safety_threshold: Threshold for considering an action unsafe
            safety_weight: Weight for the safety penalty
            device: Device to run the models on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_id = agent_id
        self.safety_threshold = safety_threshold
        self.safety_weight = safety_weight
        self.device = device
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.safety_critic = SafetyCritic(state_dim, action_dim).to(device)
        
        self.target_actor = Actor(state_dim, action_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim).to(device)
        
        # Copy weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=config.LEARNING_RATE_ACTOR
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=config.LEARNING_RATE_CRITIC
        )
        self.safety_optimizer = optim.Adam(
            self.safety_critic.parameters(), 
            lr=config.LEARNING_RATE_SAFETY
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(config.BUFFER_SIZE)
        
        # Statistics
        self.unsafe_actions_count = 0
        self.total_actions_count = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state
            explore: Whether to add exploration noise
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        self.actor.train()
        
        if explore:
            # Add exploration noise
            noise = np.random.normal(0, 0.1, size=self.action_dim)
            action = np.clip(action + noise, -1, 1)
        
        return action
    
    def is_action_unsafe(self, state: np.ndarray, action: np.ndarray, 
                         returns_history: Optional[pd.DataFrame] = None) -> bool:
        """
        Check if an action is unsafe based on the safety critic.
        
        Args:
            state: Current state
            action: Proposed action
            returns_history: Historical returns data for risk estimation
            
        Returns:
            True if the action is unsafe, False otherwise
        """
        # Use the safety critic to estimate risk
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        
        self.safety_critic.eval()
        with torch.no_grad():
            risk = self.safety_critic(state_tensor, action_tensor).cpu().data.numpy()[0, 0]
        self.safety_critic.train()
        
        # Also use the utility function for risk estimation
        if returns_history is not None:
            utility_risk = utils.estimate_risk(state, action, returns_history)
            # Combine the two risk estimates
            risk = 0.7 * risk + 0.3 * utility_risk
        
        return risk > self.safety_threshold
    
    def update(self, batch_size: int, returns_history: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Update the agent's networks based on experiences.
        
        Args:
            batch_size: Number of experiences to sample
            returns_history: Historical returns data for risk estimation
            
        Returns:
            Dictionary of loss values
        """
        if len(self.replay_buffer) < batch_size:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "safety_loss": 0.0
            }
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones, unsafes = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        unsafes_tensor = torch.FloatTensor(unsafes).unsqueeze(1).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.target_actor(next_states_tensor)
            next_q_values = self.target_critic(next_states_tensor, next_actions)
            target_q_values = rewards_tensor + (1 - dones_tensor) * config.DISCOUNT_FACTOR * next_q_values
        
        current_q_values = self.critic(states_tensor, actions_tensor)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update safety critic
        safety_predictions = self.safety_critic(states_tensor, actions_tensor)
        safety_targets = torch.FloatTensor(utils.estimate_risk_batch(states, actions, returns_history)).unsqueeze(1).to(self.device)
        safety_loss = nn.MSELoss()(safety_predictions, safety_targets)
        
        self.safety_optimizer.zero_grad()
        safety_loss.backward()
        self.safety_optimizer.step()
        
        # Update actor
        actor_actions = self.actor(states_tensor)
        actor_loss = -self.critic(states_tensor, actor_actions).mean()
        
        # Add safety penalty to actor loss
        safety_values = self.safety_critic(states_tensor, actor_actions)
        safety_penalty = self.safety_weight * torch.mean(
            torch.relu(safety_values - self.safety_threshold)
        )
        actor_loss += safety_penalty
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "safety_loss": safety_loss.item()
        }
    
    def update_target_networks(self, tau: float = 0.005):
        """
        Soft update of target networks.
        
        Args:
            tau: Interpolation parameter
        """
        # Update target actor
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Update target critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save_models(self, path: str):
        """
        Save the agent's models.
        
        Args:
            path: Path to save the models
        """
        torch.save(self.actor.state_dict(), f"{path}/actor_{self.agent_id}.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic_{self.agent_id}.pth")
        torch.save(self.safety_critic.state_dict(), f"{path}/safety_critic_{self.agent_id}.pth")
    
    def load_models(self, path: str):
        """
        Load the agent's models.
        
        Args:
            path: Path to load the models from
        """
        self.actor.load_state_dict(torch.load(f"{path}/actor_{self.agent_id}.pth"))
        self.critic.load_state_dict(torch.load(f"{path}/critic_{self.agent_id}.pth"))
        self.safety_critic.load_state_dict(torch.load(f"{path}/safety_critic_{self.agent_id}.pth"))
        
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict()) 