"""Reinforcement Learning layer for reasoning capabilities."""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class PolicyNetwork(nn.Module):
    """Policy network for action selection."""
    
    def __init__(self, input_size: int, num_actions: int, hidden_size: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to get action logits."""
        return self.network(state)


class ValueNetwork(nn.Module):
    """Value network for state value estimation."""
    
    def __init__(self, input_size: int, hidden_size: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to get state value."""
        return self.network(state)


class ReinforcementLearningLayer(nn.Module):
    """
    Reinforcement Learning layer for reasoning and decision making.
    
    This layer implements Actor-Critic architecture for policy gradient methods.
    """
    
    def __init__(
        self,
        input_size: int,
        num_actions: int,
        gamma: float = 0.99,
        use_policy_gradient: bool = True,
        use_value_network: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.use_policy_gradient = use_policy_gradient
        self.use_value_network = use_value_network
        
        # Policy network (Actor)
        if use_policy_gradient:
            self.policy_net = PolicyNetwork(input_size, num_actions)
        
        # Value network (Critic)
        if use_value_network:
            self.value_net = ValueNetwork(input_size)
            
        # Action embedding for reasoning
        self.action_embedding = nn.Embedding(num_actions, input_size)
        
    def forward(
        self,
        state: torch.Tensor,
        return_distribution: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through RL layer.
        
        Args:
            state: Input state features [batch, input_size]
            return_distribution: Whether to return action distribution
            
        Returns:
            Dictionary containing policy logits, action probabilities, 
            selected actions, and state values
        """
        result = {}
        
        # Policy network forward pass
        if self.use_policy_gradient:
            logits = self.policy_net(state)  # [batch, num_actions]
            result['logits'] = logits
            
            # Get action probabilities
            action_probs = torch.softmax(logits, dim=-1)
            result['action_probs'] = action_probs
            
            if return_distribution:
                # Sample action from distribution
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                result['action'] = action
                result['log_prob'] = dist.log_prob(action)
            else:
                # Greedy action selection
                action = torch.argmax(action_probs, dim=-1)
                result['action'] = action
        
        # Value network forward pass
        if self.use_value_network:
            state_value = self.value_net(state)  # [batch, 1]
            result['state_value'] = state_value.squeeze(-1)
        
        # Add action reasoning embedding
        if 'action' in result:
            action_embed = self.action_embedding(result['action'])
            result['action_embedding'] = action_embed
            # Combine state with action reasoning
            result['reasoned_state'] = state + action_embed
        
        return result
    
    def compute_returns(
        self,
        rewards: torch.Tensor,
        dones: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute discounted returns.
        
        Args:
            rewards: Tensor of rewards [batch, seq_len] or [batch]
            dones: Optional done flags [batch, seq_len]
            
        Returns:
            Discounted returns
        """
        # Handle 1D rewards (single step)
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
            was_1d = True
        else:
            was_1d = False
            
        returns = []
        R = 0
        
        # Reverse iterate through rewards
        for r in reversed(rewards.unbind(1)):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.stack(returns, dim=1)
        
        if was_1d:
            returns = returns.squeeze(1)
            
        return returns
    
    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
        values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute policy gradient loss.
        
        Args:
            log_probs: Log probabilities of taken actions
            returns: Computed returns
            values: Optional baseline values for advantage
            
        Returns:
            Policy loss
        """
        if values is not None:
            # Use advantage (A = R - V)
            advantages = returns - values.detach()
        else:
            advantages = returns
            
        # Policy gradient loss: -log_prob * advantage
        policy_loss = -(log_probs * advantages).mean()
        
        return policy_loss
    
    def compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute value network loss.
        
        Args:
            values: Predicted state values
            returns: Target returns
            
        Returns:
            Value loss (MSE)
        """
        value_loss = nn.functional.mse_loss(values, returns)
        return value_loss
