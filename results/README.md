# RL Drone CTF - Results & Evaluation

This folder contains evaluation scripts and presentation-ready visualizations for the RL Drone CTF project.

## Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn
```

### 2. Run Evaluation

```bash
# From the results folder
python evaluate_agents.py

# Or specify custom paths
python evaluate_agents.py /path/to/logs /path/to/output
```

### 3. View Results

The evaluation generates 6 high-quality visualizations and a comprehensive report:

**Visualizations (PNG files, 300 DPI):**
- `01_win_rate_evolution.png` - Win rate changes during training
- `02_reward_convergence.png` - Reward learning curves
- `03_episode_efficiency.png` - Episode length over time
- `04_distance_to_flag.png` - Attack team progress metrics
- `05_collision_analysis.png` - Collision patterns
- `06_performance_dashboard.png` - Complete metrics dashboard

**Reports:**
- `evaluation_report.txt` - Human-readable summary
- `metrics.json` - Machine-readable metrics

## What Each Visualization Shows

### 1. Win Rate Evolution
**For presentations:** Shows learning progress and whether teams are balanced
- Red line = Attack team win percentage
- Blue line = Defend team win percentage
- Ideal: Both lines converge near 50%

### 2. Reward Convergence
**For presentations:** Demonstrates that agents are learning
- Shows rewards over time with moving average
- Upward trend = positive learning
- Stabilization = convergence

### 3. Episode Efficiency
**For presentations:** Shows how quickly agents complete episodes
- Lower is better (faster completion)
- Decreasing trend = agents getting more efficient

### 4. Distance to Flag
**For presentations:** Key metric for attack team
- Shows average distance of attack drones to flag
- Decreasing = agents learning to approach objective

### 5. Collision Analysis
**For presentations:** Shows safety/avoidance learning
- Number of drone collisions per episode
- Decreasing = agents learning collision avoidance

### 6. Performance Dashboard
**For presentations:** One-page summary of everything
- Multiple metrics in one view
- Perfect for overview slides

## Presentation Tips

### For PowerPoint/Google Slides

1. **Import all PNGs** - They're high resolution (300 DPI)
2. **Dashboard first** - Use the dashboard as your overview slide
3. **Tell a story**: 
   - Start: "Both teams started at 0% vs 100%"
   - Middle: "After training, we see convergence..."
   - End: "Final performance shows X% win rate, indicating Y"

### Key Talking Points

**Learning Progress:**
- "The win rate evolution shows our agents learned from random behavior to strategic play"
- "Reward convergence demonstrates the RL algorithm successfully optimized behavior"

**Balance:**
- "A 50-50 win rate indicates well-balanced teams and good reward design"
- "Deviation from 50% suggests one strategy dominates"

**Efficiency:**
- "Episode length decreasing shows agents learned to complete objectives faster"
- "Collision reduction proves agents learned obstacle avoidance"

**Technical Achievement:**
- "Individual reward shaping enabled each drone to learn independently"
- "Threat-weighted rewards caused emergent coordination behavior"

## Customization

### Change Plot Style

Edit `evaluate_agents.py` line 19-24:
```python
plt.style.use('seaborn-v0_8-darkgrid')  # Try: 'ggplot', 'seaborn-v0_8-whitegrid'
plt.rcParams['figure.figsize'] = (12, 6)  # Adjust size
plt.rcParams['font.size'] = 11  # Adjust font
```

### Add More Metrics

Add new methods to the `AgentEvaluator` class:
```python
def plot_my_metric(self):
    # Your custom visualization
    pass
```

Then call it in `run_full_evaluation()`.

## Troubleshooting

**No CSV files found:**
- Make sure you've run training and generated logs
- Check the logs directory path

**Import errors:**
- Install missing packages: `pip install pandas numpy matplotlib seaborn`

**Plots look strange:**
- You may need more training episodes (recommended: 500+)
- Check data quality in CSV files

## File Structure

```
results/
├── README.md                          # This file
├── evaluate_agents.py                 # Main evaluation script
├── quick_comparison.py                # Compare multiple runs
├── presentation_template.pptx         # PowerPoint template
├── 01_win_rate_evolution.png         # Generated plots
├── 02_reward_convergence.png
├── 03_episode_efficiency.png
├── 04_distance_to_flag.png
├── 05_collision_analysis.png
├── 06_performance_dashboard.png
├── evaluation_report.txt              # Text report
└── metrics.json                       # JSON metrics
```

## For Your Class Presentation

**Recommended Slide Order:**

1. **Title Slide**
   - "RL-Based Multi-Agent Capture The Flag"

2. **Problem Overview**
   - CTF game description
   - Attack vs Defend teams
   - Why RL is suitable

3. **Approach**
   - PPO algorithm
   - Individual reward shaping
   - State representation

4. **Results Dashboard** (06_performance_dashboard.png)
   - One-slide overview of everything

5. **Learning Progress** (01_win_rate_evolution.png)
   - Show convergence from random to strategic

6. **Behavioral Metrics** (04_distance_to_flag.png & 05_collision_analysis.png)
   - Attack: approaching objective
   - Both: learning avoidance

7. **Conclusions**
   - Final win rates
   - What agents learned
   - Future improvements

## Contact

For questions about these evaluation scripts, check the main project documentation.


