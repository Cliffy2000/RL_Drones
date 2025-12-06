"""
Comprehensive Evaluation Script for RL Drone CTF Agents
Generates presentation-ready visualizations and metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


class AgentEvaluator:
    """Evaluate and visualize RL agent performance"""
    
    def __init__(self, logs_dir='../logs', output_dir='./'):
        self.logs_dir = Path(logs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        self.episodes_df = self._load_latest_csv('episodes')
        self.steps_df = self._load_latest_csv('steps')
        self.events_df = self._load_latest_csv('events')
        
        self.results = {}
    
    def _load_latest_csv(self, prefix):
        """Load the CSV file with the most data"""
        files = list(self.logs_dir.glob(f'{prefix}_*.csv'))
        if not files:
            print(f"Warning: No {prefix} CSV files found in {self.logs_dir}")
            return None
        
        # Find file with most data (not just most recent)
        best_file = None
        max_lines = 0
        for f in files:
            try:
                with open(f, 'r') as file:
                    lines = sum(1 for _ in file)
                    if lines > max_lines:
                        max_lines = lines
                        best_file = f
            except:
                continue
        
        if best_file is None or max_lines <= 1:
            print(f"Warning: No {prefix} files with data found")
            return None
        
        print(f"Loading {best_file.name} ({max_lines} rows)")
        return pd.read_csv(best_file)
    
    def calculate_metrics(self):
        """Calculate key performance metrics"""
        if self.episodes_df is None or len(self.episodes_df) == 0:
            print("No episode data available")
            return {}
        
        df = self.episodes_df
        
        # Overall metrics
        total_episodes = len(df)
        final_attack_wr = df['attack_win_rate'].iloc[-1] if len(df) > 0 else 0
        final_defend_wr = df['defend_win_rate'].iloc[-1] if len(df) > 0 else 0
        
        # Learning progress
        split_point = max(1, len(df)//4)
        early_episodes = df[:split_point]
        late_episodes = df[-split_point:]
        
        wr_improvement = late_episodes['attack_win_rate'].mean() - early_episodes['attack_win_rate'].mean() if len(df) > 1 else 0
        reward_improvement = late_episodes['attack_reward'].mean() - early_episodes['attack_reward'].mean() if len(df) > 1 else 0
        
        # Efficiency metrics
        avg_steps = df['steps'].mean()
        avg_duration = df['duration_ms'].mean() / 1000  # Convert to seconds
        
        # Reward statistics
        attack_reward_mean = df['attack_reward'].mean()
        attack_reward_std = df['attack_reward'].std()
        defend_reward_mean = df['defend_reward'].mean()
        defend_reward_std = df['defend_reward'].std()
        
        self.results = {
            'total_episodes': total_episodes,
            'final_attack_win_rate': final_attack_wr,
            'final_defend_win_rate': final_defend_wr,
            'win_rate_improvement': wr_improvement,
            'reward_improvement': reward_improvement,
            'avg_steps_per_episode': avg_steps,
            'avg_duration_seconds': avg_duration,
            'attack_reward_mean': attack_reward_mean,
            'attack_reward_std': attack_reward_std,
            'defend_reward_mean': defend_reward_mean,
            'defend_reward_std': defend_reward_std,
        }
        
        # Event statistics
        if self.events_df is not None:
            total_collisions = len(self.events_df[self.events_df['event_type'] == 'collision'])
            total_captures = len(self.events_df[self.events_df['event_type'] == 'flag_captured'])
            total_timeouts = len(self.events_df[self.events_df['event_type'] == 'time_expired'])
            
            self.results.update({
                'total_collisions': total_collisions,
                'total_flag_captures': total_captures,
                'total_timeouts': total_timeouts,
                'collisions_per_episode': total_collisions / total_episodes,
            })
        
        return self.results
    
    def plot_win_rate_evolution(self):
        """Plot 1: Win rate evolution over training"""
        if self.episodes_df is None:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        df = self.episodes_df
        ax.plot(df['episode'], df['attack_win_rate'], 
                linewidth=2, label='Attack Team (Red)', color='#e74c3c')
        ax.plot(df['episode'], df['defend_win_rate'], 
                linewidth=2, label='Defend Team (Blue)', color='#3498db')
        
        # Add 50% line
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Balance')
        
        ax.set_xlabel('Training Episode', fontweight='bold')
        ax.set_ylabel('Win Rate (%)', fontweight='bold')
        ax.set_title('Win Rate Evolution During Training', fontsize=16, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '01_win_rate_evolution.png', dpi=300, bbox_inches='tight')
        print(f"Saved: 01_win_rate_evolution.png")
        plt.close()
    
    def plot_reward_convergence(self):
        """Plot 2: Reward convergence with moving average"""
        if self.episodes_df is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        df = self.episodes_df
        window = min(50, len(df)//10)
        
        # Attack rewards
        ax1.plot(df['episode'], df['attack_reward'], alpha=0.3, color='#e74c3c')
        df['attack_ma'] = df['attack_reward'].rolling(window=window, center=True).mean()
        ax1.plot(df['episode'], df['attack_ma'], linewidth=2.5, 
                color='#c0392b', label=f'{window}-episode MA')
        ax1.set_xlabel('Episode', fontweight='bold')
        ax1.set_ylabel('Reward', fontweight='bold')
        ax1.set_title('Attack Team Reward Convergence', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Defend rewards
        ax2.plot(df['episode'], df['defend_reward'], alpha=0.3, color='#3498db')
        df['defend_ma'] = df['defend_reward'].rolling(window=window, center=True).mean()
        ax2.plot(df['episode'], df['defend_ma'], linewidth=2.5, 
                color='#2980b9', label=f'{window}-episode MA')
        ax2.set_xlabel('Episode', fontweight='bold')
        ax2.set_ylabel('Reward', fontweight='bold')
        ax2.set_title('Defend Team Reward Convergence', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '02_reward_convergence.png', dpi=300, bbox_inches='tight')
        print(f"Saved: 02_reward_convergence.png")
        plt.close()
    
    def plot_episode_efficiency(self):
        """Plot 3: Episode length (efficiency metric)"""
        if self.episodes_df is None:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        df = self.episodes_df
        window = min(50, len(df)//10)
        
        ax.scatter(df['episode'], df['steps'], alpha=0.4, s=20, color='#9b59b6')
        df['steps_ma'] = df['steps'].rolling(window=window, center=True).mean()
        ax.plot(df['episode'], df['steps_ma'], linewidth=3, 
               color='#8e44ad', label=f'{window}-episode MA')
        
        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Steps to Completion', fontweight='bold')
        ax.set_title('Episode Efficiency (Lower = Better)', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '03_episode_efficiency.png', dpi=300, bbox_inches='tight')
        print(f"Saved: 03_episode_efficiency.png")
        plt.close()
    
    def plot_distance_to_flag(self):
        """Plot 4: Average distance to flag over time"""
        if self.steps_df is None:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate per-episode averages
        episode_distances = self.steps_df.groupby('episode').agg({
            'closest_distance': 'mean',
            'avg_distance': 'mean'
        }).reset_index()
        
        window = min(30, len(episode_distances)//10)
        episode_distances['closest_ma'] = episode_distances['closest_distance'].rolling(window=window, center=True).mean()
        episode_distances['avg_ma'] = episode_distances['avg_distance'].rolling(window=window, center=True).mean()
        
        ax.plot(episode_distances['episode'], episode_distances['closest_ma'], 
               linewidth=2.5, label='Closest Drone', color='#e74c3c')
        ax.plot(episode_distances['episode'], episode_distances['avg_ma'], 
               linewidth=2.5, label='Average (All Drones)', color='#f39c12')
        
        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Distance to Flag (meters)', fontweight='bold')
        ax.set_title('Attack Team Progress: Distance to Flag', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotation
        final_dist = episode_distances['closest_ma'].iloc[-1]
        if not np.isnan(final_dist):
            ax.annotate(f'Final: {final_dist:.1f}m', 
                       xy=(episode_distances['episode'].iloc[-1], final_dist),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '04_distance_to_flag.png', dpi=300, bbox_inches='tight')
        print(f"Saved: 04_distance_to_flag.png")
        plt.close()
    
    def plot_collision_analysis(self):
        """Plot 5: Collision rate over training"""
        if self.steps_df is None:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate collisions per episode
        collisions = self.steps_df.groupby('episode')['collisions'].sum().reset_index()
        window = min(30, len(collisions)//10)
        collisions['collisions_ma'] = collisions['collisions'].rolling(window=window, center=True).mean()
        
        ax.bar(collisions['episode'], collisions['collisions'], 
              alpha=0.3, color='#e67e22', label='Per Episode')
        ax.plot(collisions['episode'], collisions['collisions_ma'], 
               linewidth=3, color='#d35400', label=f'{window}-episode MA')
        
        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Number of Collisions', fontweight='bold')
        ax.set_title('Collision Rate Over Training', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '05_collision_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved: 05_collision_analysis.png")
        plt.close()
    
    def plot_performance_summary(self):
        """Plot 6: Dashboard-style summary of key metrics"""
        if self.episodes_df is None or len(self.episodes_df) == 0:
            return
        
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        df = self.episodes_df
        
        # Win distribution pie chart
        ax1 = fig.add_subplot(gs[0, 0])
        attack_wins = df['attack_wins'].iloc[-1] if len(df) > 0 else 0
        defend_wins = df['defend_wins'].iloc[-1] if len(df) > 0 else 0
        colors = ['#e74c3c', '#3498db']
        ax1.pie([attack_wins, defend_wins], labels=['Attack', 'Defend'], 
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Final Win Distribution', fontweight='bold')
        
        # Episode length distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(df['steps'], bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax2.axvline(df['steps'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {df["steps"].mean():.0f}')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Episode Length Distribution', fontweight='bold')
        ax2.legend()
        
        # Reward distribution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.violinplot([df['attack_reward'].dropna(), df['defend_reward'].dropna()],
                      positions=[1, 2], showmeans=True, showmedians=True)
        ax3.set_xticks([1, 2])
        ax3.set_xticklabels(['Attack', 'Defend'])
        ax3.set_ylabel('Reward')
        ax3.set_title('Reward Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Win rate timeline
        ax4 = fig.add_subplot(gs[1, :])
        ax4.fill_between(df['episode'], df['attack_win_rate'], alpha=0.3, color='#e74c3c', label='Attack')
        ax4.fill_between(df['episode'], df['defend_win_rate'], alpha=0.3, color='#3498db', label='Defend')
        ax4.plot(df['episode'], df['attack_win_rate'], linewidth=2, color='#c0392b')
        ax4.plot(df['episode'], df['defend_win_rate'], linewidth=2, color='#2980b9')
        ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Episode', fontweight='bold')
        ax4.set_ylabel('Win Rate (%)', fontweight='bold')
        ax4.set_title('Win Rate Evolution', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Learning phases (early, mid, late comparison)
        ax5 = fig.add_subplot(gs[2, :2])
        n = len(df)
        phases = {
            'Early\n(0-33%)': df[:n//3],
            'Middle\n(33-66%)': df[n//3:2*n//3],
            'Late\n(66-100%)': df[2*n//3:]
        }
        
        x_pos = np.arange(len(phases))
        attack_wrs = [phases[p]['attack_win_rate'].mean() for p in phases]
        defend_wrs = [phases[p]['defend_win_rate'].mean() for p in phases]
        
        width = 0.35
        ax5.bar(x_pos - width/2, attack_wrs, width, label='Attack', color='#e74c3c')
        ax5.bar(x_pos + width/2, defend_wrs, width, label='Defend', color='#3498db')
        ax5.set_ylabel('Average Win Rate (%)', fontweight='bold')
        ax5.set_title('Win Rate by Training Phase', fontweight='bold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(phases.keys())
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Key metrics table
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        metrics_text = f"""
KEY METRICS

Episodes: {len(df)}
Avg Steps: {df['steps'].mean():.0f}
Avg Duration: {df['duration_ms'].mean()/1000:.1f}s

Attack Win Rate: {df['attack_win_rate'].iloc[-1]:.1f}%
Defend Win Rate: {df['defend_win_rate'].iloc[-1]:.1f}%

Attack Reward: {df['attack_reward'].mean():.1f}
Defend Reward: {df['defend_reward'].mean():.1f}
        """
        ax6.text(0.1, 0.5, metrics_text.strip(), fontsize=10, 
                family='monospace', verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle('RL Drone CTF: Performance Summary Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.savefig(self.output_dir / '06_performance_dashboard.png', dpi=300, bbox_inches='tight')
        print(f"Saved: 06_performance_dashboard.png")
        plt.close()
    
    def generate_report(self):
        """Generate a text report with all metrics"""
        metrics = self.calculate_metrics()
        
        report = f"""
{'='*70}
RL DRONE CTF AGENTS - EVALUATION REPORT
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

TRAINING OVERVIEW
{'-'*70}
Total Episodes Trained:        {metrics.get('total_episodes', 'N/A')}
Average Steps per Episode:     {metrics.get('avg_steps_per_episode', 0):.1f}
Average Episode Duration:      {metrics.get('avg_duration_seconds', 0):.1f} seconds

FINAL PERFORMANCE
{'-'*70}
Attack Team Win Rate:          {metrics.get('final_attack_win_rate', 0):.1f}%
Defend Team Win Rate:          {metrics.get('final_defend_win_rate', 0):.1f}%
Balance Score (closer to 0):   {abs(metrics.get('final_attack_win_rate', 50) - 50):.1f}%

LEARNING PROGRESS
{'-'*70}
Win Rate Improvement:          {metrics.get('win_rate_improvement', 0):+.1f}%
Reward Improvement:            {metrics.get('reward_improvement', 0):+.1f}

REWARD STATISTICS
{'-'*70}
Attack Team:
  Mean Reward per Episode:     {metrics.get('attack_reward_mean', 0):.2f}
  Std Deviation:               {metrics.get('attack_reward_std', 0):.2f}

Defend Team:
  Mean Reward per Episode:     {metrics.get('defend_reward_mean', 0):.2f}
  Std Deviation:               {metrics.get('defend_reward_std', 0):.2f}

EVENT STATISTICS
{'-'*70}
Total Flag Captures:           {metrics.get('total_flag_captures', 'N/A')}
Total Timeouts:                {metrics.get('total_timeouts', 'N/A')}
Total Collisions:              {metrics.get('total_collisions', 'N/A')}
Collisions per Episode:        {metrics.get('collisions_per_episode', 0):.2f}

{'='*70}
CONCLUSION:
"""
        
        # Add automatic conclusions based on metrics
        if metrics.get('final_attack_win_rate'):
            balance = abs(metrics['final_attack_win_rate'] - 50)
            if balance < 10:
                report += "\n✓ Teams are well-balanced (within 10% of 50-50 split)"
            elif balance < 20:
                report += "\n~ Teams show moderate balance (10-20% deviation)"
            else:
                report += "\n✗ Teams show significant imbalance (>20% deviation)"
        
        if metrics.get('win_rate_improvement'):
            if metrics['win_rate_improvement'] > 20:
                report += "\n✓ Strong learning progress demonstrated"
            elif metrics['win_rate_improvement'] > 10:
                report += "\n~ Moderate learning progress"
            else:
                report += "\n- Limited learning progress observed"
        
        report += f"\n\n{'='*70}\n"
        
        # Save report
        report_path = self.output_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"\nFull report saved to: {report_path}")
        
        # Also save as JSON
        json_path = self.output_dir / 'metrics.json'
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics JSON saved to: {json_path}")
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print("\n" + "="*70)
        print("STARTING COMPREHENSIVE EVALUATION")
        print("="*70 + "\n")
        
        print("Generating visualizations...")
        self.plot_win_rate_evolution()
        self.plot_reward_convergence()
        self.plot_episode_efficiency()
        self.plot_distance_to_flag()
        self.plot_collision_analysis()
        self.plot_performance_summary()
        
        print("\nGenerating report...")
        self.generate_report()
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE!")
        print("="*70)
        print(f"\nAll outputs saved to: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        print("  - 01_win_rate_evolution.png")
        print("  - 02_reward_convergence.png")
        print("  - 03_episode_efficiency.png")
        print("  - 04_distance_to_flag.png")
        print("  - 05_collision_analysis.png")
        print("  - 06_performance_dashboard.png")
        print("  - evaluation_report.txt")
        print("  - metrics.json")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    logs_dir = sys.argv[1] if len(sys.argv) > 1 else '../logs'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './'
    
    # Run evaluation
    evaluator = AgentEvaluator(logs_dir=logs_dir, output_dir=output_dir)
    evaluator.run_full_evaluation()



