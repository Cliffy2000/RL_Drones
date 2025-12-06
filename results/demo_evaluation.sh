#!/bin/bash

# Demo Evaluation Script
# Generates sample data and runs complete evaluation

echo "=========================================="
echo "RL Drone CTF - Evaluation Demo"
echo "=========================================="
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.7+."
    exit 1
fi

echo "Step 1: Installing dependencies..."
python3 -m pip install -q pandas numpy matplotlib seaborn

echo ""
echo "Step 2: Generating sample training data..."
python3 generate_sample_data.py --episodes 1000 --scenario learning

echo ""
echo "Step 3: Running evaluation..."
python3 evaluate_agents.py ../logs ./

echo ""
echo "=========================================="
echo "DEMO COMPLETE!"
echo "=========================================="
echo ""
echo "Generated visualizations:"
ls -1 *.png 2>/dev/null || echo "  (No PNG files found)"
echo ""
echo "Generated reports:"
ls -1 *.txt *.json 2>/dev/null || echo "  (No report files found)"
echo ""
echo "Open the PNG files to see presentation-ready visualizations!"


