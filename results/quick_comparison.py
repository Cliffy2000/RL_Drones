"""
Quick Comparison Script for Multiple Training Runs
Compare different hyperparameters, architectures, or training sessions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11


class RunComparator:
    """Compare multiple training runs"""
    
    def __init__(self, output_dir='./comparisons'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.runs = {}
    
    def add_run(self, name, logs_dir):
        """Add a training run to compare"""
        logs_path = Path(logs_dir)
        
        # Load latest episodes file
        files = list(logs_path.glob('episodes_*.csv'))
        if not files:
            print(f"Warning: No episode files found in {logs_dir}")
            return
        
        latest = max(files, key=lambda p: p.stat().st_mtime)
        df = pd.read_csv(latest)
        
        self.runs[name] = {
            'episodes': df,
            'final_attack_wr': df['attack_win_rate'].iloc[-1],
            'final_defend_wr': df['defend_win_rate'].iloc[-1],
            'avg_attack_reward': df['attack_reward'].mean(),
            'avg_defend_reward': df['defend_reward'].mean(),
            'total_episodes': len(df),
        }
        
        print(f"Added run: {name} ({len(df)} episodes)")
    
    def plot_win_rate_comparison(self):
        """Compare win rate evolution across runs"""
        if not self.runs:
            print("No runs to compare")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.runs)))
        
        # Attack win rates
        for (name, data), color in zip(self.runs.items(), colors):
            df = data['episodes']
            ax1.plot(df['episode'], df['attack_win_rate'], 
                    label=name, linewidth=2, color=color, alpha=0.8)
        
        ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Episode', fontweight='bold')
        ax1.set_ylabel('Attack Win Rate (%)', fontweight='bold')
        ax1.set_title('Attack Team Win Rate Comparison', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-5, 105)
        
        # Defend win rates
        for (name, data), color in zip(self.runs.items(), colors):
            df = data['episodes']
            ax2.plot(df['episode'], df['defend_win_rate'], 
                    label=name, linewidth=2, color=color, alpha=0.8)
        
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Episode', fontweight='bold')
        ax2.set_ylabel('Defend Win Rate (%)', fontweight='bold')
        ax2.set_title('Defend Team Win Rate Comparison', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-5, 105)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'win_rate_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved: win_rate_comparison.png")
        plt.close()
    
    def plot_final_performance_bars(self):
        """Bar chart comparing final performance metrics"""
        if not self.runs:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        names = list(self.runs.keys())
        x_pos = np.arange(len(names))
        
        # Final win rates
        attack_wrs = [self.runs[name]['final_attack_wr'] for name in names]
        defend_wrs = [self.runs[name]['final_defend_wr'] for name in names]
        
        width = 0.35
        ax1.bar(x_pos - width/2, attack_wrs, width, label='Attack', color='#e74c3c', alpha=0.8)
        ax1.bar(x_pos + width/2, defend_wrs, width, label='Defend', color='#3498db', alpha=0.8)
        ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Final Win Rate (%)', fontweight='bold')
        ax1.set_title('Final Win Rates', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 105)
        
        # Average rewards
        attack_rewards = [self.runs[name]['avg_attack_reward'] for name in names]
        defend_rewards = [self.runs[name]['avg_defend_reward'] for name in names]
        
        ax2.bar(x_pos - width/2, attack_rewards, width, label='Attack', color='#e74c3c', alpha=0.8)
        ax2.bar(x_pos + width/2, defend_rewards, width, label='Defend', color='#3498db', alpha=0.8)
        ax2.set_ylabel('Average Reward', fontweight='bold')
        ax2.set_title('Average Episode Rewards', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Balance score (closer to 0 is better)
        balance_scores = [abs(self.runs[name]['final_attack_wr'] - 50) for name in names]
        
        ax3.bar(x_pos, balance_scores, color='#9b59b6', alpha=0.8)
        ax3.set_ylabel('Balance Score (Lower = Better)', fontweight='bold')
        ax3.set_title('Team Balance Quality', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Good (<10%)')
        ax3.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Fair (<20%)')
        ax3.legend()
        
        # Training length
        episodes_count = [self.runs[name]['total_episodes'] for name in names]
        
        ax4.bar(x_pos, episodes_count, color='#f39c12', alpha=0.8)
        ax4.set_ylabel('Number of Episodes', fontweight='bold')
        ax4.set_title('Training Duration', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Training Runs Comparison', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison_bars.png', dpi=300, bbox_inches='tight')
        print(f"Saved: performance_comparison_bars.png")
        plt.close()
    
    def generate_comparison_table(self):
        """Create a comparison table"""
        if not self.runs:
            return
        
        # Create DataFrame
        data = []
        for name, metrics in self.runs.items():
            data.append({
                'Run Name': name,
                'Episodes': metrics['total_episodes'],
                'Attack WR (%)': f"{metrics['final_attack_wr']:.1f}",
                'Defend WR (%)': f"{metrics['final_defend_wr']:.1f}",
                'Balance': f"{abs(metrics['final_attack_wr'] - 50):.1f}",
                'Avg Attack Reward': f"{metrics['avg_attack_reward']:.1f}",
                'Avg Defend Reward': f"{metrics['avg_defend_reward']:.1f}",
            })
        
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_path = self.output_dir / 'comparison_table.csv'
        df.to_csv(csv_path, index=False)
        print(f"Saved: comparison_table.csv")
        
        # Save as formatted text
        txt_path = self.output_dir / 'comparison_table.txt'
        with open(txt_path, 'w') as f:
            f.write("TRAINING RUNS COMPARISON\n")
            f.write("=" * 100 + "\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n" + "=" * 100 + "\n")
        
        print(f"Saved: comparison_table.txt")
        print("\n" + df.to_string(index=False))
    
    def run_comparison(self):
        """Run full comparison"""
        print("\n" + "="*70)
        print("RUNNING COMPARISON ANALYSIS")
        print("="*70 + "\n")
        
        if not self.runs:
            print("No runs added. Use add_run() first.")
            return
        
        print(f"Comparing {len(self.runs)} training runs...")
        
        self.plot_win_rate_comparison()
        self.plot_final_performance_bars()
        self.generate_comparison_table()
        
        print("\n" + "="*70)
        print("COMPARISON COMPLETE!")
        print("="*70)
        print(f"\nAll outputs saved to: {self.output_dir.absolute()}")


# Example usage
if __name__ == "__main__":
    import sys
    
    comparator = RunComparator()
    
    # Example: Compare different training runs
    # comparator.add_run("Baseline", "../logs_baseline")
    # comparator.add_run("High LR", "../logs_high_lr")
    # comparator.add_run("Large Network", "../logs_large_net")
    
    # If command line arguments provided
    if len(sys.argv) > 1:
        for i in range(1, len(sys.argv), 2):
            if i + 1 < len(sys.argv):
                name = sys.argv[i]
                path = sys.argv[i + 1]
                comparator.add_run(name, path)
    else:
        print("Usage: python quick_comparison.py [name1 path1] [name2 path2] ...")
        print("\nExample:")
        print("  python quick_comparison.py 'Run 1' ../logs_run1 'Run 2' ../logs_run2")
        sys.exit(1)
    
    comparator.run_comparison()


