# Setup Instructions

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy matplotlib seaborn
```

### 2. Verify Installation

```bash
python3 -c "import pandas, numpy, matplotlib, seaborn; print('All packages installed!')"
```

## Quick Test

### Option A: Automated Demo (Recommended)

```bash
cd results/
./demo_evaluation.sh
```

This will:
1. Install dependencies
2. Generate sample training data (1000 episodes)
3. Run full evaluation
4. Create all visualizations

### Option B: Manual Test

```bash
# 1. Generate sample data
python3 generate_sample_data.py --episodes 100 --scenario learning

# 2. Run evaluation
python3 evaluate_agents.py ../logs ./

# 3. View results
ls -l *.png
cat evaluation_report.txt
```

## Directory Structure

After setup, your structure should look like:

```
Scratchpad/
├── controllers/
│   └── ctf_supervisor/
│       ├── ctf_supervisor.py
│       ├── rl_controller.py
│       └── train_rl.py
├── logs/                        # Generated during training
│   ├── episodes_*.csv
│   ├── steps_*.csv
│   └── events_*.csv
└── results/                     # THIS FOLDER
    ├── README.md
    ├── INDEX.md
    ├── SETUP.md (this file)
    ├── PRESENTATION_GUIDE.md
    ├── requirements.txt
    ├── evaluate_agents.py
    ├── generate_sample_data.py
    ├── quick_comparison.py
    ├── demo_evaluation.sh
    └── [generated PNG files after running]
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'"

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn
```

Or use pip3:
```bash
pip3 install pandas numpy matplotlib seaborn
```

### "Permission denied: ./demo_evaluation.sh"

Make it executable:
```bash
chmod +x demo_evaluation.sh
```

### "No CSV files found in ../logs"

You need to either:
1. Run training first (generates real logs)
2. Or generate sample data: `python3 generate_sample_data.py`

### Plots don't display

Plots are saved as PNG files. View them with:
- **macOS:** `open *.png`
- **Linux:** `xdg-open *.png`
- **Windows:** Explorer or your image viewer
- **Any OS:** Just open the files in your file browser

### Python version issues

Requires Python 3.7+. Check version:
```bash
python3 --version
```

If too old, install newer Python from python.org

## Next Steps

1. ✅ Install dependencies
2. ✅ Run demo to verify everything works
3. 📚 Read INDEX.md for file overview
4. 🎯 Run evaluation on your real training logs
5. 🎤 Use PRESENTATION_GUIDE.md for your class presentation

## Using with Real Training Data

After you've trained your RL agents:

```bash
# Your logs will be in logs/
# Run evaluation:
cd results/
python3 evaluate_agents.py ../logs ./

# All visualizations will be created in results/
# Use them in your presentation!
```

## Need Help?

1. Check INDEX.md for file descriptions
2. Check README.md for detailed usage
3. Check PRESENTATION_GUIDE.md for presentation tips
4. Look at the Python scripts - they have detailed comments



