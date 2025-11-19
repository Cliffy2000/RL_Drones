"""
Training script for RL-controlled drone CTF game.
Integrates PPO agent with Webots CTF supervisor.
"""

import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import json
import argparse
from typing import Dict, List, Tuple

from controller import Supervisor
from rl_controller import PPOAgent

# Setup logging
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
checkpoint_dir = Path(__file__).parent.parent.parent / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"training_{timestamp}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RLTrainer:
    """
    Trainer that integrates RL agents with Webots CTF supervisor.
    Handles state observation, action execution, and training loop.
    """
    def __init__(
        self,
        team_drone_count: int = 5,
        max_velocity: float = 1.0,
        arena_size: List[float] = [24, 15, 4],
        learning_rate: float = 3e-4,
        device: str = "cpu",
        resume_from: str = None
    ):
        # Initialize Webots supervisor
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        
        # Game parameters
        self.team_drone_count = team_drone_count
        self.max_velocity = max_velocity
        self.arena_size = arena_size
        self.flag_position = np.array([11.5, 0, 0])  # Flag on right side
        
        # Episode tracking
        self.current_episode = 0
        self.episode_time = 0
        self.max_episode_time = 300000  # 5 minutes
        self.flag_captured = False
        
        # Drone tracking
        self.spawned_drones = []
        self.active_attack_drones = set()
        self.active_defend_drones = set()
        self.prev_attack_distances = []
        self.collision_threshold = 0.5
        
        # Calculate state and action dimensions
        # State: [drone_pos(3), drone_vel(3), flag_pos(3), flag_dist(1), 
        #         team_positions(team_size*3), opponent_positions(team_size*3)]
        self.state_dim = 3 + 3 + 3 + 1 + (team_drone_count * 3) + (team_drone_count * 3)
        self.action_dim = 3  # vx, vy, vz
        
        # Initialize RL agents (one for each team)
        self.attack_agent = PPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            team_size=team_drone_count,
            lr=learning_rate,
            device=device
        )
        
        self.defend_agent = PPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            team_size=team_drone_count,
            lr=learning_rate,
            device=device
        )
        
        # Training configuration
        self.update_frequency = 2048  # Update after this many steps
        self.total_steps = 0
        self.episode_steps = 0
        self.episode_attack_reward = 0
        self.episode_defend_reward = 0
        
        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)
        
        logger.info("RLTrainer initialized")
        logger.info(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        logger.info(f"Team size: {team_drone_count}, Max velocity: {max_velocity}")
    
    def clean_leftover_drones(self):
        """Clean any leftover drones from previous runs."""
        drone_patterns = ["DRONE_ATTACK_", "DRONE_DEFEND_"]
        
        for pattern in drone_patterns:
            for i in range(self.team_drone_count):
                drone = self.supervisor.getFromDef(f"{pattern}{i}")
                if drone:
                    drone.remove()
                    logger.debug(f"Cleaned leftover {pattern}{i}")
    
    def spawn_drones(self):
        """Spawn drones for both teams."""
        import random
        import math
        
        self.spawned_drones.clear()
        self.active_attack_drones = set(range(self.team_drone_count))
        self.active_defend_drones = set(range(self.team_drone_count))
        
        root = self.supervisor.getRoot()
        children_field = root.getField("children")
        
        def get_valid_positions(x_range, count):
            positions = []
            attempts = 0
            min_spawn_dist = 2.0
            
            while len(positions) < count and attempts < 1000:
                x = random.uniform(x_range[0], x_range[1])
                y = random.uniform(-7, 7)
                z = random.uniform(1, 3)
                
                valid = True
                for pos in positions:
                    dist = math.sqrt((x-pos[0])**2 + (y-pos[1])**2 + (z-pos[2])**2)
                    if dist < min_spawn_dist:
                        valid = False
                        break
                
                if valid:
                    positions.append([x, y, z])
                attempts += 1
            
            return positions
        
        # Attack team spawns on left side
        attack_positions = get_valid_positions([-11.5, -8], self.team_drone_count)
        
        # Defend team spawns on right side
        defend_positions = get_valid_positions([8, 11.5], self.team_drone_count)
        
        # Spawn attack team (red)
        for i in range(self.team_drone_count):
            pos = attack_positions[i]
            drone_str = f'DEF DRONE_ATTACK_{i} Mavic2Pro {{ translation {pos[0]} {pos[1]} {pos[2]} bodyColor 1 0 0 }}'
            children_field.importMFNodeFromString(-1, drone_str)
            self.spawned_drones.append(self.supervisor.getFromDef(f"DRONE_ATTACK_{i}"))
        
        # Spawn defend team (blue)
        for i in range(self.team_drone_count):
            pos = defend_positions[i]
            drone_str = f'DEF DRONE_DEFEND_{i} Mavic2Pro {{ translation {pos[0]} {pos[1]} {pos[2]} bodyColor 0 0 1 }}'
            children_field.importMFNodeFromString(-1, drone_str)
            self.spawned_drones.append(self.supervisor.getFromDef(f"DRONE_DEFEND_{i}"))
        
        self.supervisor.simulationResetPhysics()
        logger.info(f"Spawned {self.team_drone_count * 2} drones for episode {self.current_episode}")
    
    def get_drone_state(self, drone_idx: int, team: str) -> np.ndarray:
        """
        Get state observation for a specific drone.
        
        Args:
            drone_idx: Index of the drone
            team: 'attack' or 'defend'
        
        Returns:
            state: numpy array of state features
        """
        drone_name = f"DRONE_{team.upper()}_{drone_idx}"
        drone = self.supervisor.getFromDef(drone_name)
        
        if not drone:
            return np.zeros(self.state_dim)
        
        # Get drone position and velocity
        position = drone.getPosition()
        velocity = drone.getVelocity()[:3]  # Linear velocity only
        
        if position is None:
            position = [0, 0, 0]
        if velocity is None:
            velocity = [0, 0, 0]
        
        # Calculate distance to flag
        flag_dist = np.linalg.norm(np.array(position) - self.flag_position)
        
        # Get team positions
        team_positions = []
        for i in range(self.team_drone_count):
            if i == drone_idx:
                continue
            teammate = self.supervisor.getFromDef(f"DRONE_{team.upper()}_{i}")
            if teammate:
                pos = teammate.getPosition()
                team_positions.extend(pos if pos else [0, 0, 0])
            else:
                team_positions.extend([0, 0, 0])
        
        # Pad if needed
        while len(team_positions) < self.team_drone_count * 3:
            team_positions.extend([0, 0, 0])
        
        # Get opponent positions
        opponent_team = 'defend' if team == 'attack' else 'attack'
        opponent_positions = []
        for i in range(self.team_drone_count):
            opponent = self.supervisor.getFromDef(f"DRONE_{opponent_team.upper()}_{i}")
            if opponent:
                pos = opponent.getPosition()
                opponent_positions.extend(pos if pos else [0, 0, 0])
            else:
                opponent_positions.extend([0, 0, 0])
        
        # Construct state vector
        state = np.concatenate([
            position,
            velocity,
            self.flag_position,
            [flag_dist],
            team_positions[:self.team_drone_count * 3],
            opponent_positions[:self.team_drone_count * 3]
        ])
        
        return state.astype(np.float32)
    
    def get_team_states(self, team: str) -> np.ndarray:
        """Get states for all drones in a team."""
        states = []
        for i in range(self.team_drone_count):
            state = self.get_drone_state(i, team)
            states.append(state)
        return np.array(states)
    
    def apply_action(self, drone_idx: int, team: str, action: np.ndarray):
        """Apply action to a drone."""
        drone_name = f"DRONE_{team.upper()}_{drone_idx}"
        drone = self.supervisor.getFromDef(drone_name)
        
        if not drone:
            return
        
        # Scale action to velocity limits
        vx = np.clip(action[0] * self.max_velocity, -self.max_velocity, self.max_velocity)
        vy = np.clip(action[1] * self.max_velocity, -self.max_velocity, self.max_velocity)
        vz = np.clip(action[2] * self.max_velocity, -self.max_velocity, self.max_velocity)
        
        # Apply velocity
        vel_field = drone.getVelocity()
        if vel_field:
            drone.setVelocity([vx, vy, vz, vel_field[3], vel_field[4], vel_field[5]])
    
    def calculate_rewards(self) -> Tuple[float, float]:
        """
        Calculate rewards for both teams based on game state.
        
        Returns:
            (blue_reward, red_reward)
        """
        blue_reward = 0
        red_reward = 0
        
        # Update episode time
        self.episode_time += self.timestep
        time_expired = self.episode_time >= self.max_episode_time
        
        # === RED TEAM (ATTACKING) REWARDS ===
        red_reward -= 1  # Time penalty
        
        # Calculate distances and progress
        current_attack_distances = []
        for i in self.active_attack_drones:
            drone = self.supervisor.getFromDef(f"DRONE_ATTACK_{i}")
            if drone:
                position = drone.getPosition()
                if position:
                    distance = np.linalg.norm(np.array(position) - self.flag_position)
                    current_attack_distances.append(distance)
                    
                    # Check for flag capture
                    if distance < 1.0 and not self.flag_captured:
                        self.flag_captured = True
                        red_reward += 250
                        blue_reward -= 250
                        logger.info(f"FLAG CAPTURED by DRONE_ATTACK_{i} at episode {self.current_episode}")
        
        # Reward for progress
        if self.prev_attack_distances and current_attack_distances:
            avg_prev = np.mean(self.prev_attack_distances)
            avg_curr = np.mean(current_attack_distances)
            if avg_curr < avg_prev:
                red_reward += 5
        
        self.prev_attack_distances = current_attack_distances
        
        # === BLUE TEAM (DEFENDING) REWARDS ===
        if not self.flag_captured:
            blue_reward += 5
        
        if time_expired and not self.flag_captured:
            blue_reward += 250
            logger.info(f"Episode {self.current_episode} timeout - defenders win!")
        
        # === COLLISION DETECTION ===
        attack_positions = []
        defend_positions = []
        
        for i in self.active_attack_drones:
            drone = self.supervisor.getFromDef(f"DRONE_ATTACK_{i}")
            if drone:
                pos = drone.getPosition()
                if pos:
                    attack_positions.append((i, np.array(pos)))
        
        for i in self.active_defend_drones:
            drone = self.supervisor.getFromDef(f"DRONE_DEFEND_{i}")
            if drone:
                pos = drone.getPosition()
                if pos:
                    defend_positions.append((i, np.array(pos)))
        
        # Check collisions
        for attack_idx, attack_pos in attack_positions:
            for defend_idx, defend_pos in defend_positions:
                distance = np.linalg.norm(attack_pos - defend_pos)
                
                if distance < self.collision_threshold:
                    red_reward -= 50
                    if attack_idx in self.active_attack_drones:
                        self.active_attack_drones.remove(attack_idx)
                        blue_reward += 50
                        logger.debug(f"Collision: DRONE_ATTACK_{attack_idx} eliminated")
        
        return blue_reward, red_reward
    
    def reset_episode(self):
        """Reset episode state."""
        # Clean up old drones
        for drone in self.spawned_drones:
            if drone:
                drone.remove()
        
        # Reset state
        self.flag_captured = False
        self.episode_time = 0
        self.episode_steps = 0
        self.episode_attack_reward = 0
        self.episode_defend_reward = 0
        self.prev_attack_distances = []
        
        # Spawn new drones
        self.spawn_drones()
        
        self.current_episode += 1
        logger.info(f"Starting episode {self.current_episode}")
    
    def train(self, max_episodes: int = 1000, save_frequency: int = 50):
        """
        Main training loop.
        
        Args:
            max_episodes: Maximum number of episodes to train
            save_frequency: Save checkpoint every N episodes
        """
        logger.info(f"Starting training for {max_episodes} episodes")
        
        # Clean and initialize
        self.clean_leftover_drones()
        self.reset_episode()
        
        try:
            while self.current_episode < max_episodes:
                # Step simulation
                if self.supervisor.step(self.timestep) == -1:
                    logger.warning("Simulation terminated")
                    break
                
                # Get states for both teams
                attack_states = self.get_team_states('attack')
                defend_states = self.get_team_states('defend')
                
                # Select actions
                attack_actions, attack_aux = self.attack_agent.select_action(attack_states)
                defend_actions, defend_aux = self.defend_agent.select_action(defend_states)
                
                # Apply actions
                for i in range(self.team_drone_count):
                    self.apply_action(i, 'attack', attack_actions[i])
                    self.apply_action(i, 'defend', defend_actions[i])
                
                # Calculate rewards
                blue_reward, red_reward = self.calculate_rewards()
                
                # Store transitions
                done = self.flag_captured or (self.episode_time >= self.max_episode_time)
                
                for i in range(self.team_drone_count):
                    if attack_aux[i][0] is not None:
                        self.attack_agent.store_transition(
                            i, attack_states[i], attack_actions[i],
                            red_reward, attack_aux[i][1], attack_aux[i][0], done
                        )
                    if defend_aux[i][0] is not None:
                        self.defend_agent.store_transition(
                            i, defend_states[i], defend_actions[i],
                            blue_reward, defend_aux[i][1], defend_aux[i][0], done
                        )
                
                # Update counters
                self.total_steps += 1
                self.episode_steps += 1
                self.episode_attack_reward += red_reward
                self.episode_defend_reward += blue_reward
                
                # Update policy
                if self.total_steps % self.update_frequency == 0:
                    logger.info(f"Updating policies at step {self.total_steps}")
                    attack_losses = self.attack_agent.update()
                    defend_losses = self.defend_agent.update()
                    
                    if attack_losses:
                        logger.info(f"Attack team losses: {attack_losses}")
                    if defend_losses:
                        logger.info(f"Defend team losses: {defend_losses}")
                
                # Check episode end
                if done:
                    logger.info(f"Episode {self.current_episode} finished:")
                    logger.info(f"  Steps: {self.episode_steps}")
                    logger.info(f"  Attack reward: {self.episode_attack_reward:.2f}")
                    logger.info(f"  Defend reward: {self.episode_defend_reward:.2f}")
                    logger.info(f"  Flag captured: {self.flag_captured}")
                    
                    # Save checkpoint
                    if self.current_episode % save_frequency == 0:
                        self.save_checkpoint()
                    
                    # Reset for next episode
                    self.reset_episode()
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
            raise
        finally:
            # Save final checkpoint
            self.save_checkpoint()
            logger.info("Training completed")
    
    def save_checkpoint(self):
        """Save training checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f"checkpoint_ep{self.current_episode}_{timestamp}.pt"
        
        metadata = {
            'episode': self.current_episode,
            'total_steps': self.total_steps,
            'timestamp': timestamp
        }
        
        self.attack_agent.save_checkpoint(
            str(checkpoint_dir / f"attack_agent_ep{self.current_episode}_{timestamp}.pt"),
            metadata
        )
        self.defend_agent.save_checkpoint(
            str(checkpoint_dir / f"defend_agent_ep{self.current_episode}_{timestamp}.pt"),
            metadata
        )
        
        # Save training config
        config = {
            'episode': self.current_episode,
            'total_steps': self.total_steps,
            'team_drone_count': self.team_drone_count,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'attack_stats': self.attack_agent.get_statistics(),
            'defend_stats': self.defend_agent.get_statistics()
        }
        
        with open(checkpoint_dir / f"training_config_ep{self.current_episode}.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Checkpoint saved at episode {self.current_episode}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        try:
            # Load agents
            attack_path = checkpoint_path.replace('.pt', '_attack.pt')
            defend_path = checkpoint_path.replace('.pt', '_defend.pt')
            
            attack_metadata = self.attack_agent.load_checkpoint(attack_path)
            defend_metadata = self.defend_agent.load_checkpoint(defend_path)
            
            self.current_episode = attack_metadata.get('episode', 0)
            self.total_steps = attack_metadata.get('total_steps', 0)
            
            logger.info(f"Resumed training from episode {self.current_episode}")
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Train RL agents for drone CTF')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--save-freq', type=int, default=50, help='Save checkpoint every N episodes')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = RLTrainer(
        learning_rate=args.lr,
        device=args.device,
        resume_from=args.resume
    )
    
    # Start training
    trainer.train(
        max_episodes=args.episodes,
        save_frequency=args.save_freq
    )


if __name__ == "__main__":
    main()

