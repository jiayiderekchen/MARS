"""
MARS: Multi-Agent Reinforcement Strategy
Multi-agent system implementation with safety critics for financial trading.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
import random
from collections import deque
import config
from agent import SafetyCriticAgent
from models import MetaController
import utils

class MultiAgentSystem:
    """
    System that manages multiple safety-critic agents.
    """
    def __init__(self, state_dim: int, action_dim: int, device: str = "cpu"):
        """
        Initialize the multi-agent system.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            device: Device to run the models on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Initialize agents
        self.agents = []
        for i in range(config.NUM_AGENTS):
            agent = SafetyCriticAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                agent_id=i,
                safety_threshold=config.SAFETY_THRESHOLDS[i],
                safety_weight=config.SAFETY_WEIGHTS[i],
                device=device
            )
            self.agents.append(agent)
        
        # Initialize meta-controller
        self.meta_controller = MetaController(state_dim, config.NUM_AGENTS).to(device)
        self.meta_optimizer = optim.Adam(
            self.meta_controller.parameters(),
            lr=config.LEARNING_RATE_META
        )
        
        # Initialize agent weights
        self.agent_weights = np.ones(config.NUM_AGENTS) / config.NUM_AGENTS
        
        # Meta-controller's replay buffer
        self.meta_replay_buffer = deque(maxlen=config.BUFFER_SIZE)
        self.meta_batch_size = config.BATCH_SIZE
        
        # Statistics
        self.episode_returns = []
        self.episode_sharpe_ratios = []
        self.episode_max_drawdowns = []
        self.unsafe_action_rates = []
    
    def add_experience_for_meta_controller(self, state: np.ndarray):
        """
        Gathers predicted Q-values and risks from each agent for the current state
        and stores them in the meta replay buffer.
        
        Args:
            state: Current normalized state
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        agent_q_values = []
        agent_risk_values = []

        for agent in self.agents:
            # Ensure networks are in eval mode for consistent predictions
            agent.actor.eval()
            agent.critic.eval()
            agent.safety_critic.eval()
            
            with torch.no_grad():
                # Get action from this agent's current policy
                hypothetical_action = agent.actor(state_tensor).cpu().data.numpy().flatten()
                hypothetical_action_tensor = torch.FloatTensor(hypothetical_action).unsqueeze(0).to(self.device)
                
                # Predicted Q-value from the agent's critic
                q_value = agent.critic(state_tensor, hypothetical_action_tensor).cpu().item()
                # Predicted risk from the agent's safety critic
                risk_value = agent.safety_critic(state_tensor, hypothetical_action_tensor).cpu().item()
            
            # Switch back to train mode
            agent.actor.train()
            agent.critic.train()
            agent.safety_critic.train()

            agent_q_values.append(q_value)
            agent_risk_values.append(risk_value)
        
        # Store the experience in the meta replay buffer
        self.meta_replay_buffer.append((
            state, 
            np.array(agent_q_values, dtype=np.float32), 
            np.array(agent_risk_values, dtype=np.float32)
        ))

    def train_meta_controller_from_buffer(self) -> float:
        """
        Samples experiences from the meta replay buffer and updates the meta-controller.
        
        Returns:
            Meta-controller loss
        """
        if len(self.meta_replay_buffer) < self.meta_batch_size:
            return 0.0  # Not enough samples to train
        
        # Sample batch from replay buffer
        batch = random.sample(self.meta_replay_buffer, self.meta_batch_size)
        states_list, q_values_list, risks_list = zip(*batch)
        
        # Convert to tensors
        states_batch = torch.FloatTensor(np.array(states_list)).to(self.device)
        q_values_batch = torch.FloatTensor(np.array(q_values_list)).to(self.device)
        risks_batch = torch.FloatTensor(np.array(risks_list)).to(self.device)
        
        # Get agent weights from meta-controller
        self.meta_controller.train()
        logits = self.meta_controller(states_batch)
        weights = F.softmax(logits, dim=1)
        
        # Calculate weighted returns and risks
        weighted_returns = torch.sum(weights * q_values_batch, dim=1)
        weighted_risks = torch.sum(weights * risks_batch, dim=1)
        
        # Calculate Sharpe-like ratio as objective
        mean_return = torch.mean(weighted_returns)
        std_return = torch.std(weighted_returns) + 1e-6  # Avoid division by zero
        mean_risk = torch.mean(weighted_risks)
        
        # Objective: maximize return/risk ratio, minimize risk
        objective = mean_return / std_return - 0.5 * mean_risk
        loss = -objective  # Minimize negative objective
        
        # Update meta-controller
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()
        
        return loss.item()
    
    def select_action(self, state: np.ndarray, explore: bool = True, 
                      returns_history: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Select an action with improved safety fallback mechanism.
        """
        # Get actions from all agents
        actions = []
        for agent in self.agents:
            action = agent.select_action(state, explore)
            actions.append(action)
        
        actions = np.array(actions)
        
        # Combine actions using weights
        combined_action = np.zeros(self.action_dim)
        for i, action in enumerate(actions):
            combined_action += self.agent_weights[i] * action
        
        # Check risk level of combined action
        if returns_history is not None:
            risk_level = utils.estimate_risk(state, combined_action, returns_history)
            
            # If risk is too high, gradually blend with safer actions
            if risk_level > 0.7:  # High risk threshold
                # Find the safest action that's below acceptable risk
                safest_action = None
                safest_risk = 1.0
                
                for i in range(config.NUM_AGENTS):
                    action_risk = utils.estimate_risk(state, actions[i], returns_history)
                    if action_risk < safest_risk:
                        safest_risk = action_risk
                        safest_action = actions[i]
                
                # Blend between combined action and safest action based on risk level
                blend_factor = min((risk_level - 0.7) / 0.3, 1.0)  # Linear blend from 0.7 to 1.0 risk
                combined_action = (1 - blend_factor) * combined_action + blend_factor * safest_action
        
        return combined_action
    
    def is_action_unsafe(self, state: np.ndarray, action: np.ndarray, 
                         returns_history: Optional[pd.DataFrame] = None) -> bool:
        """
        Check if an action is unsafe based on the most conservative agent.
        
        Args:
            state: Current state
            action: Proposed action
            returns_history: Historical returns data for risk estimation
            
        Returns:
            True if the action is unsafe, False otherwise
        """
        # Use the most conservative agent to check safety
        return self.agents[0].is_action_unsafe(state, action, returns_history)
    
    def update_agents(self, batch_size: int, returns_history: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Update all agents.
        
        Args:
            batch_size: Batch size for updates
            returns_history: Historical returns data for risk estimation
            
        Returns:
            Dictionary of average loss values
        """
        actor_losses = []
        critic_losses = []
        safety_losses = []
        
        for agent in self.agents:
            losses = agent.update(batch_size, returns_history)
            actor_losses.append(losses["actor_loss"])
            critic_losses.append(losses["critic_loss"])
            safety_losses.append(losses["safety_loss"])
            
            # Update target networks
            agent.update_target_networks()
        
        return {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "safety_loss": np.mean(safety_losses)
        }
    
    def update_meta_controller(self, states: np.ndarray, returns: np.ndarray, 
                               risks: np.ndarray) -> float:
        """
        Update the meta-controller based on agent performance.
        
        Args:
            states: Batch of states
            returns: Batch of returns for each agent
            risks: Batch of risk values for each agent
            
        Returns:
            Meta-controller loss
        """
        states_tensor = torch.FloatTensor(states).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        risks_tensor = torch.FloatTensor(risks).to(self.device)
        
        # Get agent weights from meta-controller
        logits = self.meta_controller(states_tensor)
        weights = F.softmax(logits, dim=1)
        
        # Calculate weighted returns and risks
        weighted_returns = torch.sum(weights * returns_tensor, dim=1)
        weighted_risks = torch.sum(weights * risks_tensor, dim=1)
        
        # Calculate Sharpe-like ratio as objective
        mean_return = torch.mean(weighted_returns)
        std_return = torch.std(weighted_returns) + 1e-6  # Avoid division by zero
        mean_risk = torch.mean(weighted_risks)
        
        # Objective: maximize return/risk ratio, minimize risk
        objective = mean_return / std_return - 0.5 * mean_risk
        loss = -objective  # Minimize negative objective
        
        # Update meta-controller
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()
        
        return loss.item()
    
    def update_agent_weights(self, state: np.ndarray = None):
        """
        Update the weights for combining agent actions.
        
        Args:
            state: Current state (if None, use equal weights)
        """
        if state is None:
            # Use equal weights
            self.agent_weights = np.ones(config.NUM_AGENTS) / config.NUM_AGENTS
            return
        
        # Use meta-controller to determine weights
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.meta_controller.eval()
        with torch.no_grad():
            logits = self.meta_controller(state_tensor).cpu().data.numpy()[0]
        self.meta_controller.train()
        
        # Apply softmax with temperature
        logits = logits / config.WEIGHT_UPDATE_TEMP
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        self.agent_weights = exp_logits / np.sum(exp_logits)
    
    def save_models(self, path: str):
        """
        Save all models.
        
        Args:
            path: Path to save the models
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save agent models
        for agent in self.agents:
            agent.save_models(path)
        
        # Save meta-controller
        torch.save(self.meta_controller.state_dict(), f"{path}/meta_controller.pth")
        
        # Save agent weights
        np.save(f"{path}/agent_weights.npy", self.agent_weights)
    
    def load_models(self, path: str):
        """
        Load all models.
        
        Args:
            path: Path to load the models from
        """
        # Load agent models
        for agent in self.agents:
            agent.load_models(path)
        
        # Load meta-controller
        self.meta_controller.load_state_dict(torch.load(f"{path}/meta_controller.pth"))
        
        # Load agent weights
        if os.path.exists(f"{path}/agent_weights.npy"):
            self.agent_weights = np.load(f"{path}/agent_weights.npy") 