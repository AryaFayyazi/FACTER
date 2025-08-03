"""
config.py: Centralized configuration and hyperparameters for FACTER.
"""
from pathlib import Path

class Config:
    DATASETS = {
        'ml-1m': {
            'url': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
            'paths': ['ratings.dat', 'users.dat', 'movies.dat']
        },
        'amazon': {
            'url': 'https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Movies_and_TV_5.json.gz',
            'sample_size': 2500
        }
    }
    EXTRACT_DIR = Path('./data/')
    RAW_DATA_PATH = EXTRACT_DIR / 'ml-1m'
    ALPHA = 0.2
    INITIAL_DELTA = 0.15
    MAX_NEW_TOKENS = 200
    BATCH_SIZE = 8
    MAX_ITERATIONS = 5
    PROTECTED_ATTRIBUTES = ['gender', 'age', 'occupation']
    QUANTILE_DECAY = 0.92
    VIOLATION_MEMORY_SIZE = 50
    BASE_SIMILARITY = 0.65
    MIN_SEQ_LENGTH = 5
    MAX_PROMPT_LENGTH = 2048
    SAMPLE_SIZE_PER_DATASET = 2500
    N_REFERENCE = 10
    MIN_GROUP_SIZE = 30
    N_BOOTSTRAP = 200

Config.EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
