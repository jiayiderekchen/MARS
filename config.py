"""
Configuration parameters for the Multi-Agent Reinforcement Strategy (MARS).
"""

# General parameters
SEED = 42
VERBOSE = True
SAVE_RESULTS = True
RESULT_DIR = "result/"

# Data parameters
TRAINING_DATA_PATH = "data/training_data.csv"
VALIDATION_DATA_PATH = "data/validation_data.csv"
TEST_DATA_PATH = "data/test_data.csv"
STOCK_UNIVERSE_TRAINING = "data/stock_pool_nasdaq_nyse_training.csv"
STOCK_UNIVERSE_VALIDATION = "data/stock_pool_nasdaq_nyse_validation.csv"
STOCK_UNIVERSE_TEST = "data/stock_pool_nasdaq_nyse_test.csv"
BASELINE_DATA_PATH = "data/DJI.csv"

# Market Data Processing (MDP) parameters
NUM_STOCKS = 50
INITIAL_BALANCE = 1000000  # $1M initial portfolio value
MAX_TRADE_SIZE = 150  # Maximum number of shares to trade per stock per step
TRANSACTION_COST = 0.001  # 0.1% transaction cost
DISCOUNT_FACTOR = 0.99

# Risk management parameters
MAX_POSITION_PCT = 0.2  # Maximum percentage of portfolio in any single stock
MIN_CASH_BUFFER_PCT = 0.05  # Minimum cash buffer as percentage of portfolio
MAX_LEVERAGE = 1.0  # Maximum leverage (1.0 = no leverage)
POSITION_LIMIT_ENABLED = True  # Enable position limits

# Technical indicators parameters
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
CCI_PERIOD = 20
ADX_PERIOD = 14

# Multi-agent parameters
NUM_AGENTS = 10
SAFETY_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.75, 0.85, 0.95]
SAFETY_WEIGHTS = [20.0, 15.0, 10.0, 7.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.1]

# Agent strategy descriptions
AGENT_STRATEGIES = [
    "Ultra Conservative",
    "Very Conservative", 
    "Conservative",
    "Moderately Conservative",
    "Balanced",
    "Moderately Aggressive",
    "Aggressive",
    "Very Aggressive",
    "High Risk",
    "Maximum Growth"
]

# Neural network parameters
ACTOR_HIDDEN_DIMS = [256, 128, 64]
CRITIC_HIDDEN_DIMS = [256, 128, 64]
SAFETY_CRITIC_HIDDEN_DIMS = [256, 128, 64]
LEARNING_RATE_ACTOR = 0.00005
LEARNING_RATE_CRITIC = 0.0002
LEARNING_RATE_SAFETY = 0.0002
LEARNING_RATE_META = 0.001

# Training parameters
EPISODES = 300
PATIENCE = 150  # Early stopping patience
BATCH_SIZE = 64
BUFFER_SIZE = 100000
TARGET_UPDATE_FREQ = 10
META_UPDATE_FREQ = 5  # Update meta-weights every 5 days
META_TRAIN_FREQ = 5  # Train meta-controller every 5 episodes
WEIGHT_UPDATE_TEMP = 5.0  # Temperature parameter for weight updates

# Performance evaluation parameters
PERFORMANCE_WEIGHTS = {
    'return': 0.4,      # 40% weight for returns
    'sharpe': 0.4,      # 40% weight for Sharpe ratio
    'drawdown': 0.2     # 20% weight for drawdown (penalized)
}

# Evaluation parameters
EVAL_FREQUENCY = 10  # Evaluate every 10 episodes 