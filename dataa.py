"""Legacy entry point: open the EDA notebook for the dataset inspection workflow."""
from pathlib import Path
print("Open notebooks/01_exploratory_data_analysis.ipynb in VS Code for EDA.")
print("Expected CSV:", Path(__file__).parent / "data/raw/creditcard.csv")
