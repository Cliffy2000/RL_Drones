from controller import Supervisor
import random
import math
import os
import numpy as np
import logging
from pathlib import Path

try:
    from rl_controller import PPOAgent
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("Warning: RL controller not available. Using random actions.")


TEAM_DRONE_COUNT = 5
MIN_SPAWN_DIST = 0
MAX_VELOCITY = 1
ARENA_SIZE = [5, 5, 4]


class CTFSupervisor:
    def __init__(self, use_rl=True, learning_rate=3e-4, device='cpu', 
                 attack_checkpoint=None, defend_checkpoint=None, training_mode=True):
        """
        Initialize CTF Supervisor with optional RL agents.
        
        Args:
            use_rl: Whether to use RL agents (True) or random actions (False)
            learning_rate: Learning rate for RL agents
            device: Device to use ('cpu' or 'cuda')
            attack_checkpoint: Path to attack agent checkpoint (optional)
            defend_checkpoint: Path to defend agent checkpoint (optional)
            training_mode: If True, agents will train. If False, agents use deterministic policy.
        """
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        
        # Game state tracking
        self.flag_captured = False
        self.episode_time = 0
        self.max_episode_time = 300000  # 5 minutes in ms
        self.prev_attack_distances = []
        self.active_attack_drones = set()
        self.active_defend_drones = set()
        self.collision_threshold = 0.5  # Distance threshold for collision detection
        self.flag_position = np.array([0, 0, 0])  # Flag on right side
        
        # RL configuration
        self.use_rl = use_rl and RL_AVAILABLE
        self.training_mode = training_mode
        
        # Calculate state dimensions
        # State: [drone_pos(3), drone_vel(3), flag_pos(3), flag_dist(1), 
        #         team_positions(team_size*3), opponent_positions(team_size*3)]
        self.state_dim = 3 + 3 + 3 + 1 + (TEAM_DRONE_COUNT * 3) + (TEAM_DRONE_COUNT * 3)
        self.action_dim = 3  # vx, vy, vz
        
        # Initialize RL agents if enabled
        self.attack_agent = None
        self.defend_agent = None
        
        if self.use_rl:
            print("Initializing RL agents...")
            self.attack_agent = PPOAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                team_size=TEAM_DRONE_COUNT,
                lr=learning_rate,
                device=device
            )
            
            self.defend_agent = PPOAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                team_size=TEAM_DRONE_COUNT,
                lr=learning_rate,
                device=device
            )
            
            # Load checkpoints if provided
            if attack_checkpoint:
                print(f"Loading attack agent from {attack_checkpoint}")
                self.attack_agent.load_checkpoint(attack_checkpoint)
            
            if defend_checkpoint:
                print(f"Loading defend agent from {defend_checkpoint}")
                self.defend_agent.load_checkpoint(defend_checkpoint)
            
            print(f"RL agents initialized (training_mode={training_mode})")
        else:
            print("Using random actions (RL not available or disabled)")
        
        # Training statistics
        self.episode_count = 0
        self.total_steps = 0
        self.update_frequency = 2048  # Update agents every N steps
        
        # Comprehensive tracking
        self.attack_wins = 0
        self.defend_wins = 0
        self.total_attack_reward = 0
        self.total_defend_reward = 0
        self.episode_attack_rewards = []
        self.episode_defend_rewards = []
        
        # Individual drone distance tracking for reward calculation
        self.prev_red_distances_to_flag = {}  # {drone_id: distance}
        self.prev_blue_distances_to_target = {}  # {drone_id: distance}
        
        # Setup logging
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV log files
        self.episode_log_file = log_dir / f"episodes_{timestamp}.csv"
        self.step_log_file = log_dir / f"steps_{timestamp}.csv"
        self.event_log_file = log_dir / f"events_{timestamp}.csv"
        
        # Initialize CSV files with headers
        with open(self.episode_log_file, 'w') as f:
            f.write("episode,winner,flag_captured,duration_ms,steps,attack_reward,defend_reward,attack_total,defend_total,attack_wins,defend_wins,attack_win_rate,defend_win_rate\n")
        
        with open(self.step_log_file, 'w') as f:
            f.write("episode,step,closest_distance,avg_distance,active_attack,active_defend,attack_reward,defend_reward,progress,collisions\n")
        
        with open(self.event_log_file, 'w') as f:
            f.write("episode,step,event_type,details\n")
        
        print(f"Logging initialized: {timestamp}")
        print(f"RL: {self.use_rl}, Training: {self.training_mode}")
        
        self.clean_leftover_drones()
        
    def clean_leftover_drones(self):
        drone_patterns = ["DRONE_ATTACK_", "DRONE_DEFEND_"]
        
        for pattern in drone_patterns:
            for i in range(TEAM_DRONE_COUNT):
                drone = self.supervisor.getFromDef(f"{pattern}{i}")
                if drone:
                    drone.remove()
                    print(f"Cleaned leftover {pattern}{i}")


    def start_episode(self):
        self.cleanup_episode()
        
        # Reset game state
        self.flag_captured = False
        self.episode_time = 0
        self.prev_attack_distances = []
        self.active_attack_drones = set(range(TEAM_DRONE_COUNT))
        self.active_defend_drones = set(range(TEAM_DRONE_COUNT))
        self.prev_red_distances_to_flag.clear()
        self.prev_blue_distances_to_target.clear()
        
        root = self.supervisor.getRoot()
        children_field = root.getField("children")
        
        def get_valid_positions(x_range, count):
            positions = []
            attempts = 0
            while len(positions) < count and attempts < 1000:
                x = random.uniform(x_range[0], x_range[1])
                y = random.uniform(-2, 2)  # Within arena bounds (-7.5 to 7.5)
                z = random.uniform(1, 3)  # 1 to 3 height
                
                # Check minimum distance from existing positions
                valid = True
                for pos in positions:
                    dist = math.sqrt((x-pos[0])**2 + (y-pos[1])**2 + (z-pos[2])**2)
                    if dist < MIN_SPAWN_DIST:
                        valid = False
                        break
                
                if valid:
                    positions.append([x, y, z])
                attempts += 1
            
            return positions
        
        # Attack team spawns on left side (x: -12 to -8)
        attack_positions = get_valid_positions([-4, -3], TEAM_DRONE_COUNT)
        
        # Defend team spawns on right side (x: 8 to 12)
        defend_positions = get_valid_positions([3, 4], TEAM_DRONE_COUNT)
        
        self.flag_position = get_valid_positions([0, 4], 1)[0]
        flag = self.supervisor.getFromDef("FLAG")
        if flag:
            flag.getField("translation").setSFVec3f(self.flag_position)
        else:
            print("Couldn't set flag!")
        # Spawn attack team (red)
        for i in range(TEAM_DRONE_COUNT):
            pos = attack_positions[i]
            drone_str = f'DEF DRONE_ATTACK_{i} Mavic2Pro {{ translation {pos[0]} {pos[1]} {pos[2]} bodyColor 1 0 0 }}'
            children_field.importMFNodeFromString(-1, drone_str)
        
        # Spawn defend team (blue)
        for i in range(TEAM_DRONE_COUNT):
            pos = defend_positions[i]
            drone_str = f'DEF DRONE_DEFEND_{i} Mavic2Pro {{ translation {pos[0]} {pos[1]} {pos[2]} bodyColor 0 0 1 }}'
            children_field.importMFNodeFromString(-1, drone_str)

        self.supervisor.simulationResetPhysics()
        
    
    def get_drone_state(self, drone_idx, team):
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
            return np.zeros(self.state_dim, dtype=np.float32)
        
        # Get drone position and velocity
        position = drone.getPosition()
        velocity = drone.getVelocity()[:3] if drone.getVelocity() else [0, 0, 0]
        
        if position is None:
            position = [0, 0, 0]
        
        # Calculate distance to flag
        flag_dist = np.linalg.norm(np.array(position) - self.flag_position)
        
        # Get team positions
        team_positions = []
        for i in range(TEAM_DRONE_COUNT):
            if i == drone_idx:
                continue
            teammate = self.supervisor.getFromDef(f"DRONE_{team.upper()}_{i}")
            if teammate:
                pos = teammate.getPosition()
                team_positions.extend(pos if pos else [0, 0, 0])
            else:
                team_positions.extend([0, 0, 0])
        
        # Pad if needed
        while len(team_positions) < TEAM_DRONE_COUNT * 3:
            team_positions.extend([0, 0, 0])
        
        # Get opponent positions
        opponent_team = 'defend' if team == 'attack' else 'attack'
        opponent_positions = []
        for i in range(TEAM_DRONE_COUNT):
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
            team_positions[:TEAM_DRONE_COUNT * 3],
            opponent_positions[:TEAM_DRONE_COUNT * 3]
        ])
        
        return state.astype(np.float32)
    
    def get_team_states(self, team):
        """Get states for all drones in a team."""
        states = []
        for i in range(TEAM_DRONE_COUNT):
            state = self.get_drone_state(i, team)
            states.append(state)
        return np.array(states)
    
    def get_actions(self):
        """
        Get actions for both teams.
        Uses RL agents if available, otherwise random actions.
        
        Returns:
            Dictionary with 'attack' and 'defend' action lists
            If training_mode, also returns auxiliary data (log_probs, values)
        """
        if self.use_rl and self.attack_agent and self.defend_agent:
            # Get states for both teams
            attack_states = self.get_team_states('attack')
            defend_states = self.get_team_states('defend')
            
            # Get actions from RL agents
            deterministic = not self.training_mode
            attack_actions, attack_aux = self.attack_agent.select_action(
                attack_states, deterministic=deterministic
            )
            defend_actions, defend_aux = self.defend_agent.select_action(
                defend_states, deterministic=deterministic
            )
            
            # Scale actions to velocity range
            attack_actions = attack_actions * MAX_VELOCITY
            defend_actions = defend_actions * MAX_VELOCITY
            
            result = {
                'attack': attack_actions.tolist(),
                'defend': defend_actions.tolist()
            }
            
            if self.training_mode:
                result['attack_aux'] = attack_aux
                result['defend_aux'] = defend_aux
                result['attack_states'] = attack_states
                result['defend_states'] = defend_states
            
            return result
        else:
            # Random actions as fallback
            return {
                'attack': [[random.uniform(-MAX_VELOCITY, MAX_VELOCITY), 
                       random.uniform(-MAX_VELOCITY, MAX_VELOCITY), 
                       random.uniform(-MAX_VELOCITY, MAX_VELOCITY)] 
                       for _ in range(TEAM_DRONE_COUNT)],
                'defend': [[random.uniform(-MAX_VELOCITY, MAX_VELOCITY), 
                        random.uniform(-MAX_VELOCITY, MAX_VELOCITY), 
                        random.uniform(-MAX_VELOCITY, MAX_VELOCITY)] 
                        for _ in range(TEAM_DRONE_COUNT)]
            }
    
    def control_drone(self, drone, velocity):
        if not drone:
            return

        orientation = drone.getOrientation()
        up_vector = [orientation[1], orientation[4], orientation[7]]  # [R01, R11, R21]
        # Desired up vector
        desired_up = [0, 1, 0]

        velocity = np.maximum(np.minimum(velocity, MAX_VELOCITY), -MAX_VELOCITY)
        torque = np.cross(up_vector, desired_up) * 5.0 
        
        vel_field = drone.getVelocity()
        if vel_field:
            drone.setVelocity([velocity[0], velocity[1], velocity[2], torque[0], torque[1], torque[2]])  
    
    def apply_actions(self, actions):
        for i in range(TEAM_DRONE_COUNT):
            drone = self.supervisor.getFromDef(f"DRONE_ATTACK_{i}")
            if drone:
                self.control_drone(drone, actions['attack'][i])
        
        for i in range(TEAM_DRONE_COUNT):
            drone = self.supervisor.getFromDef(f"DRONE_DEFEND_{i}")
            if drone:
                self.control_drone(drone, actions['defend'][i]) 
    
    
    def calculate_rewards(self):
        """
        Individual reward calculation for each drone based on their behavior.
        
        RED TEAM (Attack):
        - Dense distance-based progress rewards toward flag
        - Danger zone penalty when near blue drones
        - Timestep penalty to encourage speed
        - Large reward for flag capture
        
        BLUE TEAM (Defend):
        - Threat-weighted pursuit rewards (target red drones closest to flag)
        - Proximity bonuses for being near high-threat targets
        - Elimination rewards
        - Terminal rewards for timeout win/loss
        
        Returns:
            (red_rewards_list, blue_rewards_list): Individual rewards for each drone
        """
        # Initialize individual reward arrays
        red_rewards = [0.0] * TEAM_DRONE_COUNT
        blue_rewards = [0.0] * TEAM_DRONE_COUNT
        
        # Constants
        MAX_DIST = 25.0  # Approximate max distance in arena
        DANGER_ZONE_DIST = 2.0
        INTERCEPTION_ZONE_DIST = 3.0
        FLAG_CAPTURE_DIST = 1.0
        
        # Update episode time
        self.episode_time += self.timestep
        time_expired = self.episode_time >= self.max_episode_time
        
        # === COLLECT DRONE POSITIONS ===
        red_positions = {}  # {drone_id: np.array([x, y, z])}
        blue_positions = {}
        
        for i in range(TEAM_DRONE_COUNT):
            red_drone = self.supervisor.getFromDef(f"DRONE_ATTACK_{i}")
            if red_drone and i in self.active_attack_drones:
                pos = red_drone.getPosition()
                if pos:
                    red_positions[i] = np.array(pos)
            
            blue_drone = self.supervisor.getFromDef(f"DRONE_DEFEND_{i}")
            if blue_drone and i in self.active_defend_drones:
                pos = blue_drone.getPosition()
                if pos:
                    blue_positions[i] = np.array(pos)
        
        # === RED TEAM REWARDS ===
        min_distance_to_flag = float('inf')
        closest_red_drone_id = None
        
        for red_id, red_pos in red_positions.items():
            # Base timestep penalty
            red_rewards[red_id] -= 0.2
            
            # Calculate distance to flag
            dist_to_flag = np.linalg.norm(red_pos - self.flag_position)
            
            if dist_to_flag < min_distance_to_flag:
                min_distance_to_flag = dist_to_flag
                closest_red_drone_id = red_id
            
            # Dense progress reward (distance-based shaping)
            if red_id in self.prev_red_distances_to_flag:
                prev_dist = self.prev_red_distances_to_flag[red_id]
                progress = prev_dist - dist_to_flag
                red_rewards[red_id] += progress * 2.0
            
            self.prev_red_distances_to_flag[red_id] = dist_to_flag
            
            # Danger zone penalty (when near blue drones)
            for blue_pos in blue_positions.values():
                dist_to_enemy = np.linalg.norm(red_pos - blue_pos)
                if dist_to_enemy < DANGER_ZONE_DIST:
                    red_rewards[red_id] -= 1.0
            
            # Check for flag capture
            if dist_to_flag < FLAG_CAPTURE_DIST and not self.flag_captured:
                self.flag_captured = True
                red_rewards[red_id] += 300  # Capturer gets large reward
                
                # Team bonus for other red drones
                for other_red_id in red_positions.keys():
                    if other_red_id != red_id:
                        red_rewards[other_red_id] += 50
                
                # Penalty for all blue drones
                for blue_id in blue_positions.keys():
                    blue_rewards[blue_id] -= 80
                
                # Log event
                with open(self.event_log_file, 'a') as f:
                    f.write(f"{self.episode_count},{self.total_steps},flag_captured,drone_{red_id}_dist_{dist_to_flag:.2f}\n")
        
        # === BLUE TEAM REWARDS ===
        
        # Find highest-threat red drone (closest to flag) for each blue drone to target
        red_threats = []  # [(red_id, dist_to_flag, position)]
        for red_id, red_pos in red_positions.items():
            dist_to_flag = np.linalg.norm(red_pos - self.flag_position)
            red_threats.append((red_id, dist_to_flag, red_pos))
        
        red_threats.sort(key=lambda x: x[1])  # Sort by distance to flag (closest first)
        
        for blue_id, blue_pos in blue_positions.items():
            # Each blue drone targets the red drone closest to the flag
            if red_threats:
                target_red_id, target_dist_to_flag, target_pos = red_threats[0]
                
                # Calculate threat weight (closer to flag = higher threat)
                threat_weight = 1.0 + (MAX_DIST - target_dist_to_flag) / MAX_DIST
                threat_weight = min(threat_weight, 2.0)  # Cap at 2.0
                
                # Distance to target
                dist_to_target = np.linalg.norm(blue_pos - target_pos)
                
                # Pursuit reward (moving toward high-threat target)
                if blue_id in self.prev_blue_distances_to_target:
                    prev_dist = self.prev_blue_distances_to_target[blue_id]
                    pursuit_progress = prev_dist - dist_to_target
                    blue_rewards[blue_id] += pursuit_progress * threat_weight
                
                self.prev_blue_distances_to_target[blue_id] = dist_to_target
                
                # Proximity bonus (being close to high-threat targets)
                if dist_to_target < INTERCEPTION_ZONE_DIST and threat_weight > 1.5:
                    blue_rewards[blue_id] += 1.0
        
        # === COLLISION DETECTION & ELIMINATION REWARDS ===
        collisions_this_step = 0
        eliminated_red_drones = []
        
        for red_id, red_pos in list(red_positions.items()):
            for blue_id, blue_pos in blue_positions.items():
                dist = np.linalg.norm(red_pos - blue_pos)
                
                if dist < self.collision_threshold:
                    collisions_this_step += 1
                    
                    # Penalty for red drone
                    red_rewards[red_id] -= 50
                    
                    # Reward for blue drone that caused elimination
                    blue_rewards[blue_id] += 60
                    
                    # Remove red drone
                    if red_id in self.active_attack_drones:
                        self.active_attack_drones.remove(red_id)
                        eliminated_red_drones.append(red_id)
                        # This is important to ensure the drone is removed from simulation
                        drone_node = self.supervisor.getFromDef(f"DRONE_ATTACK_{red_id}")
                        if drone_node:
                            drone_node.remove()
                            print(f"Physically removed drone DRONE_ATTACK_{red_id} from simulation")
     
                    # Log event
                    with open(self.event_log_file, 'a') as f:
                        f.write(f"{self.episode_count},{self.total_steps},collision,attack_{red_id}_vs_defend_{blue_id}\n")
                    
                    break  # Only count one collision per red drone
        
        # === TERMINAL REWARDS ===
        
        # Timeout win for blue team
        if time_expired and not self.flag_captured:
            for blue_id in blue_positions.keys():
                blue_rewards[blue_id] += 80
            
            # Log event
            with open(self.event_log_file, 'a') as f:
                f.write(f"{self.episode_count},{self.total_steps},time_expired,defenders_win\n")
        
        # === LOGGING ===
        
        # Calculate average distance for logging
        avg_distance = np.mean([np.linalg.norm(pos - self.flag_position) for pos in red_positions.values()]) if red_positions else 0
        
        # Track progress (for logging)
        progress_made = False
        if self.prev_attack_distances:
            avg_prev = np.mean(self.prev_attack_distances)
            avg_curr = np.mean([np.linalg.norm(pos - self.flag_position) for pos in red_positions.values()]) if red_positions else 0
            progress_made = avg_curr < avg_prev
        
        self.prev_attack_distances = [np.linalg.norm(pos - self.flag_position) for pos in red_positions.values()]
        
        # Log step metrics to CSV every 50 steps
        if self.total_steps % 50 == 0:
            avg_red = np.mean([r for r in red_rewards if r != 0]) if any(r != 0 for r in red_rewards) else 0
            avg_blue = np.mean([r for r in blue_rewards if r != 0]) if any(r != 0 for r in blue_rewards) else 0
            
            with open(self.step_log_file, 'a') as f:
                f.write(f"{self.episode_count},{self.total_steps},{min_distance_to_flag:.2f},{avg_distance:.2f},{len(self.active_attack_drones)},{len(self.active_defend_drones)},{avg_red:.2f},{avg_blue:.2f},{1 if progress_made else 0},{collisions_this_step}\n")
        
        return red_rewards, blue_rewards 
    
    
    def cleanup_episode(self):
        for drone_id in self.active_attack_drones:
            drone = self.supervisor.getFromDef(f"DRONE_ATTACK_{drone_id}")
            if drone:
                drone.remove()

        for drone_id in self.active_defend_drones:
            drone = self.supervisor.getFromDef(f"DRONE_DEFEND_{drone_id}")
            if drone:
                drone.remove()

        self.active_attack_drones.clear()
        self.active_defend_drones.clear()
    
    
    def save_checkpoint(self, episode=None):
        """Save RL agent checkpoints."""
        if not self.use_rl or not self.attack_agent or not self.defend_agent:
            print("Warning: Cannot save checkpoint - RL not enabled")
            return
        
        checkpoint_dir = Path(__file__).parent.parent.parent / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ep_str = f"ep{episode if episode else self.episode_count}"
        
        metadata = {
            'episode': episode if episode else self.episode_count,
            'total_steps': self.total_steps,
            'timestamp': timestamp
        }
        
        attack_path = checkpoint_dir / f"attack_agent_{ep_str}_{timestamp}.pt"
        defend_path = checkpoint_dir / f"defend_agent_{ep_str}_{timestamp}.pt"
        
        self.attack_agent.save_checkpoint(str(attack_path), metadata)
        self.defend_agent.save_checkpoint(str(defend_path), metadata)
        
        print(f"Checkpoints saved: {ep_str}")
    
    def run(self, max_episodes=None, save_frequency=50):
        """
        Main control loop with optional RL training.
        
        Args:
            max_episodes: Maximum episodes to run (None = infinite)
            save_frequency: Save checkpoint every N episodes
        """
        self.start_episode()
        episode_steps = 0
        episode_attack_reward = 0
        episode_defend_reward = 0
        
        print(f"Starting: training={self.training_mode}, rl={self.use_rl}, episodes={max_episodes or 'inf'}")
        
        while self.supervisor.step(self.timestep) != -1:
            # Get actions (with optional RL)
            action_data = self.get_actions()
            
            # Apply actions
            self.apply_actions(action_data)
            
            # Calculate rewards (returns individual rewards for each drone)
            red_rewards, blue_rewards = self.calculate_rewards()
            
            # Store transitions if in training mode
            if self.training_mode and self.use_rl:
                done = self.flag_captured or (self.episode_time >= self.max_episode_time)
                
                # Store transitions for each drone
                if 'attack_aux' in action_data:
                    attack_states = action_data['attack_states']
                    attack_actions = np.array(action_data['attack']) / MAX_VELOCITY  # Normalize back
                    
                    for i in range(TEAM_DRONE_COUNT):
                        log_prob, value = action_data['attack_aux'][i]
                        if log_prob is not None:
                            self.attack_agent.store_transition(
                                i, attack_states[i], attack_actions[i],
                                red_rewards[i], value, log_prob, done
                            )
                
                if 'defend_aux' in action_data:
                    defend_states = action_data['defend_states']
                    defend_actions = np.array(action_data['defend']) / MAX_VELOCITY  # Normalize back
                    
                    for i in range(TEAM_DRONE_COUNT):
                        log_prob, value = action_data['defend_aux'][i]
                        if log_prob is not None:
                            self.defend_agent.store_transition(
                                i, defend_states[i], defend_actions[i],
                                blue_rewards[i], value, log_prob, done
                            )
            
            # Update counters
            self.total_steps += 1
            episode_steps += 1
            episode_attack_reward += sum(red_rewards)
            episode_defend_reward += sum(blue_rewards)
            
            # Update RL agents periodically
            if self.training_mode and self.use_rl and self.total_steps % self.update_frequency == 0:
                print(f"Updating agents at step {self.total_steps}")
                attack_losses = self.attack_agent.update()
                defend_losses = self.defend_agent.update()
                
                if attack_losses:
                    print(f"  Attack - PL:{attack_losses['policy_loss']:.4f} VL:{attack_losses['value_loss']:.4f} E:{attack_losses['entropy']:.4f}")
                if defend_losses:
                    print(f"  Defend - PL:{defend_losses['policy_loss']:.4f} VL:{defend_losses['value_loss']:.4f} E:{defend_losses['entropy']:.4f}")
            
            # Check for episode end
            if self.flag_captured or self.episode_time >= self.max_episode_time:
                self.episode_count += 1
                
                # Determine winner
                if self.flag_captured:
                    self.attack_wins += 1
                    winner = "🔴 ATTACK (Red Team)"
                else:
                    self.defend_wins += 1
                    winner = "🔵 DEFEND (Blue Team)"
                
                # Update total rewards
                self.total_attack_reward += episode_attack_reward
                self.total_defend_reward += episode_defend_reward
                self.episode_attack_rewards.append(episode_attack_reward)
                self.episode_defend_rewards.append(episode_defend_reward)
                
                # Calculate win rates
                attack_win_rate = (self.attack_wins / self.episode_count) * 100
                defend_win_rate = (self.defend_wins / self.episode_count) * 100
                
                # Log episode to CSV
                with open(self.episode_log_file, 'a') as f:
                    f.write(f"{self.episode_count},{winner.split()[0]},{1 if self.flag_captured else 0},{self.episode_time},{episode_steps},{episode_attack_reward:.2f},{episode_defend_reward:.2f},{self.total_attack_reward:.2f},{self.total_defend_reward:.2f},{self.attack_wins},{self.defend_wins},{attack_win_rate:.2f},{defend_win_rate:.2f}\n")
                
                # Minimal console output
                print(f"Ep {self.episode_count}: {winner.split()[0]} | Steps:{episode_steps} | R:{episode_attack_reward:+.0f}/{episode_defend_reward:+.0f} | Wins:{self.attack_wins}/{self.defend_wins}")
                
                # Save checkpoint periodically
                if self.training_mode and self.use_rl and self.episode_count % save_frequency == 0:
                    self.save_checkpoint(self.episode_count)
                    print(f"Checkpoint saved: episode {self.episode_count}")
                
                # Check if max episodes reached
                if max_episodes and self.episode_count >= max_episodes:
                    print(f"\nTraining complete: {max_episodes} episodes")
                    print(f"Steps:{self.total_steps} | AWins:{self.attack_wins}({attack_win_rate:.1f}%) | DWins:{self.defend_wins}({defend_win_rate:.1f}%)")
                    print(f"Avg rewards: A:{self.total_attack_reward/self.episode_count:+.1f} D:{self.total_defend_reward/self.episode_count:+.1f}")
                    
                    if self.use_rl:
                        self.save_checkpoint(self.episode_count)
                    break
                
                # Reset for next episode
                episode_steps = 0
                episode_attack_reward = 0
                episode_defend_reward = 0
                self.start_episode()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CTF Drone Supervisor with RL')
    parser.add_argument('--use-rl', action='store_true', default=True, 
                       help='Use RL agents (default: True)')
    parser.add_argument('--no-rl', dest='use_rl', action='store_false',
                       help='Disable RL and use random actions')
    parser.add_argument('--training', action='store_true', default=True,
                       help='Enable training mode (default: True)')
    parser.add_argument('--inference', dest='training', action='store_false',
                       help='Disable training mode (inference only)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use (default: cpu)')
    parser.add_argument('--attack-checkpoint', type=str, default=None,
                       help='Path to attack agent checkpoint')
    parser.add_argument('--defend-checkpoint', type=str, default=None,
                       help='Path to defend agent checkpoint')
    parser.add_argument('--max-episodes', type=int, default=None,
                       help='Maximum number of episodes (default: infinite)')
    parser.add_argument('--save-freq', type=int, default=50,
                       help='Save checkpoint every N episodes (default: 50)')
    
    args = parser.parse_args()
    
    # Create supervisor with configuration
    ctf = CTFSupervisor(
        use_rl=args.use_rl,
        learning_rate=args.lr,
        device=args.device,
        attack_checkpoint=args.attack_checkpoint,
        defend_checkpoint=args.defend_checkpoint,
        training_mode=args.training
    )
    
    # Run simulation
    ctf.run(
        max_episodes=args.max_episodes,
        save_frequency=args.save_freq
    )
