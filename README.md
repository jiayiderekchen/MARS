# MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management

## 🚀 Overview

We introduce the source code of our paper "[MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management](https://)". 

## 🏗️ Key Features

- ✅ **Multi-Agent Coordination**: Dynamic agent weighting based on market conditions
- ✅ **Safety-First Design**: Built-in safety critics prevent excessive risk-taking
- ✅ **Adaptive Strategy**: Meta-controller learns optimal agent combinations
- ✅ **Risk Management**: Configurable position limits and safety thresholds
- ✅ **Technical Indicators**: Integration of RSI, MACD, CCI, ADX indicators
- ✅ **Multiple Markets**: Support for DJI, HSI, and custom datasets


## 🛠️ Installation


### Setup

1. Clone the repository:
```bash
git clone https://github.com/jiayiderekchen/MARS.git
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
