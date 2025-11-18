
"""
config.py

Central configuration file for CardioSegNet.

"""

from pathlib import Path

DATA_ROOT = Path(r"F:\Datasets\ACDC_preprocessed")  

# ACDC training directory paths 
SLICES_DIR = DATA_ROOT / "ACDC_training_slices"
VOLUMES_DIR = DATA_ROOT / "ACDC_training_volumes"
TEST_VOLUMES_DIR = DATA_ROOT / "ACDC_testing_volumes"

# Project root = folder that contains config.py
PROJECT_ROOT = Path(__file__).resolve().parent

# Results directories (models, logs, figures...)
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = RESULTS_DIR / "models"
LOG_DIR = RESULTS_DIR / "logs"
FIG_DIR = RESULTS_DIR / "figures"

# Make sure these directories exist at import time (harmless if they already exist)
for d in (RESULTS_DIR, MODEL_DIR, LOG_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Basic training hyperparameters for quick prototyping
IMG_SIZE = 128            # Baseline: 128x128; will upgrade to 256x256 in Phase 2
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 20
VAL_SPLIT = 0.1           # Fraction of samples used for validation
RANDOM_SEED = 42
