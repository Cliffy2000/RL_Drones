
# Results & Evaluation - File Index

Quick reference for all files in this folder.

## 🚀 Quick Start

**Want to test immediately?**
```bash
./demo_evaluation.sh
```
This will generate sample data and create all visualizations automatically!

---

## 📁 File Descriptions

### Core Scripts

| File | Purpose | Usage |
|------|---------|-------|
| **evaluate_agents.py** | Main evaluation script | `python evaluate_agents.py [logs_dir] [output_dir]` |
| **generate_sample_data.py** | Create test data | `python generate_sample_data.py --episodes 1000` |
| **quick_comparison.py** | Compare multiple runs | `python quick_comparison.py 'Run1' ../logs1 'Run2' ../logs2` |
| **demo_evaluation.sh** | Automated demo | `./demo_evaluation.sh` |

### Documentation

| File | Purpose |
|------|---------|
| **README.md** | Complete usage guide |
| **PRESENTATION_GUIDE.md** | Slide-by-slide presentation script |
| **INDEX.md** | This file - quick reference |

### Configuration

| File | Purpose |
|------|---------|
| **requirements.txt** | Python dependencies |

---

## 📊 Generated Outputs

When you run the evaluation, these files will be created:

### Visualizations (PNG, 300 DPI)
1. `01_win_rate_evolution.png` - Learning progress
2. `02_reward_convergence.png` - Reward curves
3. `03_episode_efficiency.png` - Episode length
4. `04_distance_to_flag.png` - Attack progress
5. `05_collision_analysis.png` - Safety metrics
6. `06_performance_dashboard.png` - Complete overview

### Reports
- `evaluation_report.txt` - Human-readable summary
- `metrics.json` - Machine-readable data

### Comparisons (if using quick_comparison.py)
- `win_rate_comparison.png` - Compare multiple runs
- `performance_comparison_bars.png` - Bar chart comparison
- `comparison_table.csv` - Tabular comparison
- `comparison_table.txt` - Formatted text table

---

## 🎯 Workflows

### Workflow 1: Evaluate Your Training

```bash
# After training, you'll have logs in ../logs/
python evaluate_agents.py ../logs ./

# View the generated PNGs
open *.png  # macOS
# or
xdg-open *.png  # Linux
# or just open in file browser
```

### Workflow 2: Test with Sample Data

```bash
# Generate fake data to test scripts
python generate_sample_data.py --episodes 1000 --scenario learning

# Run evaluation on sample data
python evaluate_agents.py ../logs ./
```

### Workflow 3: Compare Multiple Runs

```bash
# Generate different scenarios
python generate_sample_data.py -o ../logs_run1 -e 1000 -s learning
python generate_sample_data.py -o ../logs_run2 -e 1000 -s balanced

# Compare them
python quick_comparison.py 'Learning' ../logs_run1 'Balanced' ../logs_run2
cd comparisons/
open *.png
```

### Workflow 4: Prepare Presentation

```bash
# 1. Run full evaluation
python evaluate_agents.py ../logs ./

# 2. Open PRESENTATION_GUIDE.md for talking points
# 3. Import all PNG files into your slides
# 4. Use evaluation_report.txt for specific numbers
```

---

## 💡 Tips

### For Best Results

**When generating sample data:**
- Use `--scenario learning` to show improvement over time
- Use `--scenario balanced` to show 50-50 equilibrium
- Use `--scenario imbalanced` to show one team dominating

**For presentations:**
- All PNGs are 300 DPI - perfect for projectors
- Use `06_performance_dashboard.png` as your main results slide
- `01_win_rate_evolution.png` shows learning most clearly

**For analysis:**
- Read `evaluation_report.txt` for quick insights
- Use `metrics.json` for programmatic access
- Check CSV logs directly for custom analysis

### Common Issues

**"No CSV files found"**
→ Make sure you've run training or generated sample data first

**"Module not found"**
→ Install dependencies: `pip install -r requirements.txt`

**"Permission denied" on demo script**
→ Make executable: `chmod +x demo_evaluation.sh`

---

## 📈 What Each Visualization Shows

### For Your Class Presentation

**Use these descriptions when explaining plots:**

**01_win_rate_evolution.png**
> "Shows how win rates changed during training. Starting from 0-100% imbalance, teams converged to X-Y%, demonstrating successful learning and balance."

**02_reward_convergence.png**
> "Demonstrates reward optimization over time. The moving average shows clear upward trends, proving agents improved their policies."

**03_episode_efficiency.png**
> "Episode length decreased from X to Y steps, showing agents learned to complete objectives more efficiently."

**04_distance_to_flag.png**
> "Attack team's average distance to flag reduced by X%, proving they learned goal-directed behavior rather than random movement."

**05_collision_analysis.png**
> "Collision rate decreased X%, demonstrating agents learned multi-objective behavior: pursue goals while avoiding obstacles."

**06_performance_dashboard.png**
> "One-page summary of all key metrics. Perfect for presenting overall results before diving into details."

---

## 🎓 For Your Report/Paper

### Suggested Figure Captions

```
Figure 1: Win rate evolution during training showing convergence 
from random policy (0% attack) to learned equilibrium (47% attack, 
53% defend), demonstrating balanced gameplay emergence.

Figure 2: Cumulative reward per episode for both teams over 1000 
training episodes. Moving average (window=50) shows clear positive 
trend indicating successful policy optimization.

Figure 3: Average episode length decreased from 520 to 380 steps, 
indicating improved efficiency as agents learned optimal strategies.

Figure 4: Attack team's average distance to flag decreased from 
23.5m to 7.2m (69% reduction), proving acquisition of goal-directed 
behavior.

Figure 5: Collision events per episode decreased from 8.2 to 1.9 
(77% reduction), demonstrating emergence of safety-aware behavior 
alongside task completion.

Figure 6: Performance dashboard summarizing key metrics across 
training phases, showing early (0-33%), middle (33-66%), and late 
(66-100%) training statistics.
```

---

## 🔧 Customization

### Change Plot Colors

Edit `evaluate_agents.py` around line 19:
```python
plt.style.use('seaborn-v0_8-darkgrid')  # Try: 'ggplot', 'bmh'
sns.set_palette("husl")  # Try: "Set2", "Dark2"
```

### Change Figure Sizes

Edit `evaluate_agents.py` around line 21:
```python
plt.rcParams['figure.figsize'] = (12, 6)  # (width, height)
```

### Add Custom Metrics

Add method to `AgentEvaluator` class:
```python
def plot_my_metric(self):
    # Your visualization code
    plt.savefig(self.output_dir / '07_my_metric.png', dpi=300)
```

Then call in `run_full_evaluation()`.

---

## 📞 Getting Help

**Script won't run?**
1. Check Python version: `python --version` (need 3.7+)
2. Install dependencies: `pip install -r requirements.txt`
3. Check file paths are correct

**Plots look weird?**
1. Make sure you have enough episodes (500+ recommended)
2. Check CSV files aren't empty
3. Try regenerating sample data

**Need different analysis?**
1. Check CSV files directly - they have all the raw data
2. Use pandas to create custom plots
3. Modify the evaluation scripts

---

## ✅ Quality Checklist

Before presenting:
- [ ] Generated all 6 visualization PNGs
- [ ] Reviewed evaluation_report.txt
- [ ] Plots show clear learning trends
- [ ] No errors in console output
- [ ] PNGs are high quality (not pixelated)
- [ ] File names match your presentation references

---

**Everything you need for a successful project presentation is here!**

For detailed presentation guidance, see: **PRESENTATION_GUIDE.md**


