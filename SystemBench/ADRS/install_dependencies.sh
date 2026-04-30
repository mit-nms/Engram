#!/bin/bash

# Install dependencies for ADRS environments
echo "Installing dependencies for ADRS environments..."

# Install graphviz (required by cloudcast)
echo "Installing graphviz..."
pip install graphviz

# Install other common dependencies that might be needed
echo "Installing other common dependencies..."
pip install networkx matplotlib seaborn pandas numpy scipy colorama

echo "Dependencies installed successfully!"
echo ""
echo "You can now run the ADRS wrapper with real evaluations."
echo "Example: bash scripts/paper.sh"
