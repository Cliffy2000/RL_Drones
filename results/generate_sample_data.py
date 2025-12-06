"""
Generate Sample Training Data for Testing Evaluation Scripts
Creates realistic-looking CSV files that simulate actual training logs
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def generate_sample_training_data(output_dir='../logs', num_episodes=1000, scenario='learning'):
    """
    Generate sample CSV files that look like real training data
    
    Args:
        output_dir: Where to save the CSV files
        num_episodes: How many episodes to simulate
        scenario: 'learning' (agents improve), 'balanced' (50-50), or 'imbalanced' (one dominates)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Generating {num_episodes} episodes of sample data ({scenario} scenario)...")
    
    # Initialize tracking variables
    attack_wins = 0
    defend_wins = 0
    
    episodes_data = []
    steps_data = []
    events_data = []
    
    step_counter = 0
    
    for episode in range(1, num_episodes + 1):
        # Simulate learning progress
        progress = episode / num_episodes
        
        if scenario == 'learning':
            # Attack team learns to approach flag
            base_attack_skill = 0.0 + progress * 0.5  # 0% to 50%
            base_defend_skill = 0.5 - progress * 0.1  # 50% to 40%
        elif scenario == 'balanced':
            # Both teams equally matched
            base_attack_skill = 0.48
            base_defend_skill = 0.52
        else:  # imbalanced
            # One team dominates
            base_attack_skill = 0.7
            base_defend_skill = 0.3
        
        # Add noise
        attack_win_prob = np.clip(base_attack_skill + np.random.normal(0, 0.1), 0, 1)
        
        # Determine winner
        flag_captured = np.random.random() < attack_win_prob
        
        if flag_captured:
            attack_wins += 1
            winner = "ATTACK"
            duration_ms = np.random.randint(2000, 15000)  # Shorter when captured
            steps = duration_ms // 32  # Assuming 32ms timestep
        else:
            defend_wins += 1
            winner = "DEFEND"
            duration_ms = np.random.randint(20000, 30000)  # Full duration
            steps = duration_ms // 32
        
        # Calculate win rates
        attack_win_rate = (attack_wins / episode) * 100
        defend_win_rate = (defend_wins / episode) * 100
        
        # Simulate rewards (improving over time)
        if flag_captured:
            attack_reward = np.random.normal(150 + progress * 200, 50)  # Improves
            defend_reward = np.random.normal(-100, 30)
        else:
            attack_reward = np.random.normal(-200 + progress * 150, 40)  # Less negative
            defend_reward = np.random.normal(200, 50)
        
        # Cumulative totals
        attack_total = sum([e['attack_reward'] for e in episodes_data]) + attack_reward
        defend_total = sum([e['defend_reward'] for e in episodes_data]) + defend_reward
        
        # Episode data
        episodes_data.append({
            'episode': episode,
            'winner': winner,
            'flag_captured': 1 if flag_captured else 0,
            'duration_ms': duration_ms,
            'steps': steps,
            'attack_reward': attack_reward,
            'defend_reward': defend_reward,
            'attack_total': attack_total,
            'defend_total': defend_total,
            'attack_wins': attack_wins,
            'defend_wins': defend_wins,
            'attack_win_rate': attack_win_rate,
            'defend_win_rate': defend_win_rate,
        })
        
        # Simulate steps data (every 50 steps)
        num_step_logs = steps // 50
        initial_distance = np.random.uniform(20, 25)
        
        for step_log in range(num_step_logs):
            step_counter += 50
            
            # Distance decreases as agents learn
            distance_reduction = progress * 15  # Learn to get closer
            closest_distance = max(1, initial_distance - distance_reduction - step_log * 0.5)
            avg_distance = closest_distance + np.random.uniform(2, 5)
            
            # Active drones (some collisions)
            active_attack = max(3, 5 - step_log // 3)
            active_defend = 5
            
            # Rewards per step
            attack_step_reward = np.random.normal(progress * 2 - 0.5, 1)
            defend_step_reward = np.random.normal(0.3, 0.5)
            
            # Progress and collisions
            progress_made = 1 if np.random.random() < progress else 0
            collisions = np.random.poisson(0.3 * (1 - progress * 0.8))  # Fewer as learning
            
            steps_data.append({
                'episode': episode,
                'step': step_counter,
                'closest_distance': closest_distance,
                'avg_distance': avg_distance,
                'active_attack': active_attack,
                'active_defend': active_defend,
                'attack_reward': attack_step_reward,
                'defend_reward': defend_step_reward,
                'progress': progress_made,
                'collisions': collisions,
            })
        
        # Simulate events
        # Collisions
        num_collisions = np.random.poisson(3 * (1 - progress * 0.7))  # Fewer collisions over time
        for _ in range(num_collisions):
            collision_step = np.random.randint(1, steps)
            events_data.append({
                'episode': episode,
                'step': step_counter - steps + collision_step,
                'event_type': 'collision',
                'details': f'attack_{np.random.randint(0,5)}_vs_defend_{np.random.randint(0,5)}'
            })
        
        # Flag capture or timeout
        if flag_captured:
            events_data.append({
                'episode': episode,
                'step': step_counter,
                'event_type': 'flag_captured',
                'details': f'drone_{np.random.randint(0,5)}_dist_{np.random.uniform(0.5, 1.0):.2f}'
            })
        else:
            events_data.append({
                'episode': episode,
                'step': step_counter,
                'event_type': 'time_expired',
                'details': 'defenders_win'
            })
    
    # Save CSVs
    episodes_df = pd.DataFrame(episodes_data)
    steps_df = pd.DataFrame(steps_data)
    events_df = pd.DataFrame(events_data)
    
    episodes_path = output_path / f'episodes_{timestamp}.csv'
    steps_path = output_path / f'steps_{timestamp}.csv'
    events_path = output_path / f'events_{timestamp}.csv'
    
    episodes_df.to_csv(episodes_path, index=False)
    steps_df.to_csv(steps_path, index=False)
    events_df.to_csv(events_path, index=False)
    
    print(f"\nGenerated files:")
    print(f"  - {episodes_path.name} ({len(episodes_df)} rows)")
    print(f"  - {steps_path.name} ({len(steps_df)} rows)")
    print(f"  - {events_path.name} ({len(events_df)} rows)")
    
    print(f"\nFinal statistics:")
    print(f"  Attack wins: {attack_wins} ({attack_win_rate:.1f}%)")
    print(f"  Defend wins: {defend_wins} ({defend_win_rate:.1f}%)")
    print(f"  Total steps: {step_counter}")
    
    return episodes_path.parent


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample training data')
    parser.add_argument('--output', '-o', default='../logs', help='Output directory')
    parser.add_argument('--episodes', '-e', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--scenario', '-s', choices=['learning', 'balanced', 'imbalanced'], 
                       default='learning', help='Training scenario')
    
    args = parser.parse_args()
    
    logs_dir = generate_sample_training_data(
        output_dir=args.output,
        num_episodes=args.episodes,
        scenario=args.scenario
    )
    
    print(f"\n{'='*70}")
    print("Sample data generated successfully!")
    print(f"{'='*70}")
    print(f"\nTo evaluate, run:")
    print(f"  python evaluate_agents.py {logs_dir}")


