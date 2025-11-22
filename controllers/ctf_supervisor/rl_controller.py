import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import json


# Setup logging
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"rl_training_{timestamp}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for continuous action space control.
    Actor outputs mean and std for action distribution.
    Critic outputs state value estimate.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Actions between -1 and 1
        )
        
        # Log std is a learnable parameter
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        logger.info(f"Initialized ActorCriticNetwork with state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        features = self.shared(state)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std)
        value = self.critic(features)
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        """Get action from policy."""
        action_mean, action_std, value = self.forward(state)
        
        if deterministic:
            return action_mean, None, value
        
        # Sample from normal distribution
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value
    
    def evaluate_actions(self, state, action):
        """Evaluate actions for training."""
        action_mean, action_std, value = self.forward(state)
        
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, value, entropy


class RolloutBuffer:
    """Buffer to store experience during rollouts."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self):
        """Get all stored data as tensors."""
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.FloatTensor(np.array(self.actions)),
            torch.FloatTensor(np.array(self.rewards)).unsqueeze(-1),
            torch.FloatTensor(np.array(self.values)),
            torch.FloatTensor(np.array(self.log_probs)),
            torch.FloatTensor(np.array(self.dones)).unsqueeze(-1)
        )
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.rewards)


class PPOAgent:
    """
    Proximal Policy Optimization agent for drone control.
    Implements PPO algorithm without gym wrapper.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        team_size: int = 5,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.team_size = team_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        
        # Create networks for each drone
        self.networks = nn.ModuleList([
            ActorCriticNetwork(state_dim, action_dim).to(self.device)
            for _ in range(team_size)
        ])
        
        # Optimizers for each network
        self.optimizers = [
            optim.Adam(net.parameters(), lr=lr)
            for net in self.networks
        ]
        
        # Rollout buffers for each drone
        self.buffers = [RolloutBuffer() for _ in range(team_size)]
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_iterations = 0
        
        logger.info(f"Initialized PPOAgent with {team_size} drones")
        logger.info(f"Hyperparameters: lr={lr}, gamma={gamma}, clip_epsilon={clip_epsilon}")
    
    def select_action(self, states: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, List]:
        """
        Select actions for all drones in the team.
        
        Args:
            states: Array of shape (team_size, state_dim)
            deterministic: Whether to use deterministic policy
        
        Returns:
            actions: Array of shape (team_size, action_dim)
            aux_data: List of (log_prob, value) tuples for each drone
        """
        actions = []
        aux_data = []
        
        with torch.no_grad():
            for i, (network, state) in enumerate(zip(self.networks, states)):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, log_prob, value = network.get_action(state_tensor, deterministic)
                
                actions.append(action.cpu().numpy()[0])
                if not deterministic:
                    aux_data.append((
                        log_prob.cpu().numpy()[0],
                        value.cpu().numpy()[0]
                    ))
                else:
                    aux_data.append((None, value.cpu().numpy()[0]))
        
        return np.array(actions), aux_data
    
    def store_transition(self, drone_idx: int, state, action, reward, value, log_prob, done):
        """Store a transition for a specific drone."""
        self.buffers[drone_idx].add(state, action, reward, value, log_prob, done)
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Tensor of shape (steps, 1)
            values: Tensor of shape (steps, 1)
            dones: Tensor of shape (steps, 1)
            next_value: Scalar tensor
        
        Returns:
            advantages: Tensor of shape (steps, 1)
            returns: Tensor of shape (steps, 1)
        """
        advantages = []
        gae = 0
        
        # Compute advantages in reverse
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value_step = next_value
            else:
                next_value_step = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_value_step * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        
        # advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = torch.FloatTensor(advantages).unsqueeze(-1).to(self.device)
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, epochs: int = 10, batch_size: int = 64):
        """
        Update policy using PPO algorithm.
        
        Args:
            epochs: Number of epochs to train
            batch_size: Batch size for training
        """
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_count = 0
        
        # Update each drone's policy
        for drone_idx, (network, optimizer, buffer) in enumerate(
            zip(self.networks, self.optimizers, self.buffers)
        ):
            if len(buffer) == 0:
                continue
            
            # Get data from buffer
            states, actions, rewards, values, old_log_probs, dones = buffer.get()
            
            # Move to device
            states = states.to(self.device)
            actions = actions.to(self.device)
            old_log_probs = old_log_probs.to(self.device)
            
            # Compute advantages and returns
            with torch.no_grad():
                # Get value of last state for bootstrapping
                last_value = values[-1]
                advantages, returns = self.compute_gae(rewards, values, dones, last_value)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update for multiple epochs
            dataset_size = len(states)
            indices = np.arange(dataset_size)
            
            for epoch in range(epochs):
                np.random.shuffle(indices)
                
                for start_idx in range(0, dataset_size, batch_size):
                    end_idx = min(start_idx + batch_size, dataset_size)
                    batch_indices = indices[start_idx:end_idx]
                    
                    # Get batch
                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]
                    
                    # Evaluate actions
                    log_probs, state_values, entropy = network.evaluate_actions(
                        batch_states, batch_actions
                    )
                    
                    # Compute ratio for PPO
                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    
                    # Clipped surrogate objective
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    value_loss = F.mse_loss(state_values, batch_returns)
                    
                    # Entropy bonus
                    entropy_loss = -entropy.mean()
                    
                    # Total loss
                    loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                    
                    # Optimize
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(network.parameters(), self.max_grad_norm)
                    optimizer.step()
                    
                    # Track losses
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()
                    update_count += 1
            
            # Clear buffer
            buffer.clear()
        
        self.training_iterations += 1
        
        # Log training statistics
        if update_count > 0:
            avg_policy_loss = total_policy_loss / update_count
            avg_value_loss = total_value_loss / update_count
            avg_entropy = total_entropy / update_count
            
            logger.info(f"Update {self.training_iterations}: "
                       f"Policy Loss: {avg_policy_loss:.4f}, "
                       f"Value Loss: {avg_value_loss:.4f}, "
                       f"Entropy: {avg_entropy:.4f}")
            
            return {
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss,
                'entropy': avg_entropy
            }
        
        return None
    
    def save_checkpoint(self, filepath: str, metadata: dict = None):
        """Save model checkpoint."""
        checkpoint = {
            'networks': [net.state_dict() for net in self.networks],
            'optimizers': [opt.state_dict() for opt in self.optimizers],
            'training_iterations': self.training_iterations,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'team_size': self.team_size,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef,
            }
        }
        
        if metadata:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            for net, state_dict in zip(self.networks, checkpoint['networks']):
                net.load_state_dict(state_dict)
            
            for opt, state_dict in zip(self.optimizers, checkpoint['optimizers']):
                opt.load_state_dict(state_dict)
            
            self.training_iterations = checkpoint.get('training_iterations', 0)
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            self.episode_lengths = checkpoint.get('episode_lengths', [])
            
            logger.info(f"Checkpoint loaded from {filepath}")
            logger.info(f"Resumed at iteration {self.training_iterations}")
            
            return checkpoint.get('metadata', {})
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {filepath}: {e}")
            raise
    
    def get_statistics(self) -> Dict:
        """Get training statistics."""
        if len(self.episode_rewards) > 0:
            return {
                'episodes': len(self.episode_rewards),
                'mean_reward': np.mean(self.episode_rewards[-100:]),
                'std_reward': np.std(self.episode_rewards[-100:]),
                'mean_length': np.mean(self.episode_lengths[-100:]),
                'training_iterations': self.training_iterations
            }
        return {}

