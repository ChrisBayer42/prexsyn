#!/bin/bash
# Launch script for PrexSyn Jupyter Notebook

echo "🔬 Launching PrexSyn Jupyter Notebook..."

# Check if we're in the right directory
if [ ! -f "PrexSyn_Interactive_Exploration.ipynb" ]; then
    echo "❌ Error: Not in the PrexSyn directory or notebook not found"
    echo "Please run this script from /home/christopher/git/prexsyn"
    exit 1
fi

# Source conda and activate environment
export PATH="/home/christopher/miniconda/bin:$PATH"
source /home/christopher/miniconda/etc/profile.d/conda.sh
conda activate prexsyn

echo "✅ Environment activated!"
echo "📦 Starting Jupyter Notebook..."
echo "💻 Your browser should open automatically, or visit the URL shown below"

# Start Jupyter Notebook
jupyter notebook PrexSyn_Interactive_Exploration.ipynb 
    --ip=0.0.0.0 
    --port=8888 
    --no-browser 
    --NotebookApp.token='' 
    --NotebookApp.password=''

echo "🎉 Jupyter Notebook launched successfully!"
