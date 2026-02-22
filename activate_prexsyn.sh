#!/bin/bash
# Activation script for PrexSyn conda environment

echo "🔬 Activating PrexSyn conda environment..."

# Check if we're in the right directory
if [ ! -d "." ] || [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Not in the PrexSyn directory"
    echo "Please run this script from /home/christopher/git/prexsyn"
    exit 1
fi

# Source conda and activate environment
export PATH="/home/christopher/miniconda/bin:$PATH"
source /home/christopher/miniconda/etc/profile.d/conda.sh
conda activate prexsyn

echo "✅ PrexSyn conda environment activated!"
echo "📦 Python version: $(python --version)"
echo "💻 You can now use PrexSyn commands"
echo "🔬 Try: python -c 'import prexsyn; print(\"PrexSyn version:\", prexsyn.__version__)'"

# Start a shell
bash