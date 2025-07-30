# MARS: Multi-Agent Reinforcement Strategy

## 🚀 Overview

MARS (Multi-Agent Reinforcement Strategy) is a sophisticated multi-agent reinforcement learning framework designed for algorithmic trading. The system employs multiple safety-critic agents managed by a meta-controller to make dynamic trading decisions across diverse market conditions.

## 🏗️ Architecture

### Core Components

- **Multi-Agent System**: Multiple safety-critic agents with different risk tolerances
- **Meta-Controller**: Neural network that dynamically weights agent contributions
- **Safety Critics**: Specialized networks that evaluate action safety
- **Actor-Critic Framework**: Deep reinforcement learning architecture for decision making
- **Experience Replay**: Buffer system for efficient learning from historical experiences

### Key Features

- ✅ **Multi-Agent Coordination**: Dynamic agent weighting based on market conditions
- ✅ **Safety-First Design**: Built-in safety critics prevent excessive risk-taking
- ✅ **Adaptive Strategy**: Meta-controller learns optimal agent combinations
- ✅ **Risk Management**: Configurable position limits and safety thresholds
- ✅ **Technical Indicators**: Integration of RSI, MACD, CCI, ADX indicators
- ✅ **Multiple Markets**: Support for DJI, HSI, and custom datasets

## 📊 Performance

The MARS system has been tested on multiple financial markets:

- **Dow Jones Industrial Average (DJI)**: 2020-2024 data
- **Hang Seng Index (HSI)**: 2020-2024 data  
- **NASDAQ/NYSE**: Selected stock universes

Key metrics tracked:
- Cumulative returns
- Sharpe ratio
- Maximum drawdown
- Win rate
- Risk-adjusted performance

## 🛠️ Installation

### Prerequisites

```bash
Python 3.8+
PyTorch 1.9+
NumPy
Pandas
Matplotlib
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MARS.git
cd MARS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Data is included**: The repository includes pre-processed financial datasets:
   - Training, validation, and test datasets (2018-2024)
   - DJI baseline data for comparison
   - Stock universe configuration files

## 🚀 Quick Start

### Step 1: Train the Model

```bash
python main.py --mode train 
```

### Step 2: Evaluate Performance

```bash
python main.py --mode test --model_path result/best_models/
```

### Step 3: Train and Test Together

```bash
python main.py --mode both
```

## ⚙️ Configuration

Key parameters in `config.py`:

```python
# Agent Configuration
NUM_AGENTS = 10
SAFETY_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.75, 0.85, 0.95]
SAFETY_WEIGHTS = [20.0, 15.0, 10.0, 7.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.1]

# Training Parameters
EPISODES = 300
BATCH_SIZE = 64
LEARNING_RATE_ACTOR = 0.00005
LEARNING_RATE_CRITIC = 0.0002

# Trading Parameters
INITIAL_BALANCE = 1000000
MAX_TRADE_SIZE = 150
TRANSACTION_COST = 0.001
MAX_POSITION_PCT = 0.2
```

## 📈 Usage Examples

### Basic Training and Testing

```python
# Train the model
python main.py --mode train

# Test the trained model
python main.py --mode test --model_path result/best_models/

# Train and test in one command
python main.py --mode both
```

### Custom Training with Specific Parameters

```python
from multi_agent import MultiAgentSystem
import config

# Initialize system
multi_agent = MultiAgentSystem(
    state_dim=config.STATE_DIM,
    action_dim=config.NUM_STOCKS,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Load included training data
train_data = pd.read_csv("data/training_data.csv", index_col=0, parse_dates=True)
multi_agent.train(train_data, episodes=500)
```

### Backtesting

```python
# Load trained model
multi_agent.load_models("result/best_models/")

# Run backtest on included test data
test_data = pd.read_csv("data/test_data.csv", index_col=0, parse_dates=True)
results = multi_agent.evaluate(test_data)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

## 📊 Results Analysis

The system generates comprehensive analysis including:

- **Performance Plots**: Cumulative returns, drawdown analysis
- **Agent Weight Evolution**: How meta-controller adapts agent weights
- **Risk Metrics**: Detailed risk-return analysis
- **Trading Statistics**: Win rates, holding periods, turnover

Results are saved in the `result/` directory:
```
result/
├── training_curves.png
├── test_results.png
├── agent_weights_evolution.png
├── test_results_detailed.csv
└── models/
```

## 🧪 Experiments

The framework supports various experimental configurations using the included datasets:

1. **MARS Main**: Full multi-agent system with adaptive weighting
2. **MARS Ablation Static**: Fixed equal agent weights  
3. **MARS Ablation Homogeneous**: Single agent type only
4. **Baseline**: Traditional single-agent approaches
5. **Sensitivity Analysis**: Agent count variations

Run experiments using:
```bash
python main.py --mode train    # Standard training
python main.py --mode test     # Testing only
python main.py --mode both     # Complete training and testing
```

## 📚 Technical Details

### State Representation

The system uses a comprehensive state representation including:
- Price and volume data
- Technical indicators (RSI, MACD, CCI, ADX)
- Portfolio positions and cash balance
- Market volatility measures
- Historical performance metrics

### Action Space

Actions represent position changes for each stock:
- Continuous action space [-1, 1] per stock
- Positive values indicate buying, negative indicate selling
- Actions are scaled by maximum trade size limits

### Reward Function

Multi-objective reward combining:
- Portfolio returns
- Risk-adjusted metrics
- Safety penalties
- Transaction costs

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use MARS in your research, please cite:

```bibtex
@article{mars2024,
  title={MARS: Multi-Agent Reinforcement Strategy for Algorithmic Trading},
  author={Your Name},
  journal={Journal of Financial Technology},
  year={2024}
}
```

## 🆘 Support

- **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/yourusername/MARS/issues)
- **Discussions**: Join community discussions in [GitHub Discussions](https://github.com/yourusername/MARS/discussions)
- **Email**: Contact us at your.email@domain.com

## 🏆 Acknowledgments

- Built with PyTorch and inspired by modern multi-agent RL research
- Includes curated financial datasets for immediate use
- Pre-processed market data covering multiple time periods (2018-2024)
- Thanks to the open-source community for various utility libraries

---

**⭐ If you find MARS useful, please consider giving it a star!** 