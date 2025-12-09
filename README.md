# MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management

## 🚀 Overview

We introduce the source code of our paper "[MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management](https://arxiv.org/pdf/2508.01173?)". 

![Preview](images/MARS_framework_5.5.drawio.png)

The MARS framework architecture. The system processes the Market State ($s_t$) through two parallel components. The Meta-Adaptive Controller (MAC) produces agent weights ($w_t$), while the Heterogeneous Agent Ensemble (HAE) generates proposed actions ($a_t^i$). These outputs are aggregated and passed through a Risk Management Overlay to produce the final executed action ($A'_t$).

## 🏗️ Key Features

- **Hierarchical Two-Tiered Architecture**: Implements a high-level Meta-Adaptive Controller (MAC) that learns to dynamically orchestrate a low-level ensemble of trading agents, allowing the system to adapt its strategy to different market regimes (bull vs. bear).
- **Heterogeneous Agent Ensemble (HAE)**: Moves beyond a single-agent paradigm by employing an ensemble of agents, each with a unique and explicit risk profile (e.g., conservative, aggressive). This leverages behavioral diversity for more robust decision-making.
- **Explicit Risk Management with Safety-Critic**: Each agent is equipped with a dedicated Safety-Critic network. This allows risk to be proactively managed as part of the learning objective, rather than being a reactive penalty, leading to superior capital preservation.
- **State-of-the-Art Performance**: Achieves superior risk-adjusted returns compared to established DRL baselines. Demonstrates exceptional performance in minimizing drawdowns and volatility, particularly during bear markets.
- **Modular and Extensible**: The framework is designed with modular components, allowing for easy experimentation with different numbers of agents, risk profiles, and meta-controller strategies.


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

## 📖 Citation

If you use MARS in your research, please cite:

```bibtex
@article{chen2025mars,
  title={MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management},
  author={Chen, Jiayi and Li, Jing and Wang, Guiling},
  journal={arXiv preprint arXiv:2508.01173},
  year={2025}
}
```
