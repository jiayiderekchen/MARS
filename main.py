"""
MARS: Multi-Agent Reinforcement Strategy
Main training and evaluation pipeline for the multi-agent trading system.
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import argparse
from typing import Dict, List, Tuple, Optional
import config
import utils
from multi_agent import MultiAgentSystem

def train(multi_agent: MultiAgentSystem, train_data: pd.DataFrame, 
          validation_data: pd.DataFrame, returns_history: pd.DataFrame):
    """
    Train the multi-agent system.
    
    Args:
        multi_agent: Multi-agent system to train
        train_data: Training data
        validation_data: Validation data
        returns_history: Historical returns data
    """
    print("Starting training...")
    
    # Create result directory
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    
    # Training statistics
    train_returns = []
    train_sharpe_ratios = []
    train_drawdowns = []
    validation_returns = []
    validation_sharpe_ratios = []
    validation_drawdowns = []
    validation_composite_scores = []  # Track composite performance scores
    
    # Early stopping variables
    best_composite_score = float('-inf')
    patience_counter = 0
    best_episode = 0
    
    # Training loop
    for episode in range(config.EPISODES):
        start_time = time.time()
        
        # Train on training data
        train_stats = run_episode(
            multi_agent=multi_agent,
            data=train_data,
            returns_history=returns_history,
            train=True,
            batch_size=config.BATCH_SIZE
        )
        
        train_returns.append(train_stats["return"])
        train_sharpe_ratios.append(train_stats["sharpe"])
        train_drawdowns.append(train_stats["max_drawdown"])
        
        # Evaluate on validation data
        if (episode + 1) % config.EVAL_FREQUENCY == 0:
            validation_stats = run_episode(
                multi_agent=multi_agent,
                data=validation_data,
                returns_history=returns_history,
                train=False,
                batch_size=0
            )
            
            validation_returns.append(validation_stats["return"])
            validation_sharpe_ratios.append(validation_stats["sharpe"])
            validation_drawdowns.append(validation_stats["max_drawdown"])
            
            # Calculate composite performance score
            composite_score = utils.calculate_composite_performance_score(
                return_val=validation_stats["return"],
                sharpe_ratio=validation_stats["sharpe"],
                max_drawdown=validation_stats["max_drawdown"]
            )
            validation_composite_scores.append(composite_score)
            
            # Check for best performance and early stopping
            if composite_score > best_composite_score:
                best_composite_score = composite_score
                best_episode = episode + 1
                patience_counter = 0
                print(f"  🎯 New best composite score: {composite_score:.4f} (Return: {validation_stats['return']:.4f}, "
                      f"Sharpe: {validation_stats['sharpe']:.4f}, Max DD: {validation_stats['max_drawdown']:.4f})")
                multi_agent.save_models(f"{config.RESULT_DIR}/best_models")
            else:
                patience_counter += config.EVAL_FREQUENCY
                print(f"  📊 Composite score: {composite_score:.4f} (No improvement for {patience_counter} episodes)")
            
            # Print statistics
            print(f"Episode {episode + 1}/{config.EPISODES} - Time: {time.time() - start_time:.2f}s")
            print(f"  Train - Return: {train_stats['return']:.4f}, Sharpe: {train_stats['sharpe']:.4f}, Max DD: {train_stats['max_drawdown']:.4f}")
            print(f"  Valid - Return: {validation_stats['return']:.4f}, Sharpe: {validation_stats['sharpe']:.4f}, Max DD: {validation_stats['max_drawdown']:.4f}")
            
            # Early stopping check
            if patience_counter >= config.PATIENCE:
                print(f"\n🛑 Early stopping triggered! No improvement for {config.PATIENCE} episodes.")
                print(f"Best composite score: {best_composite_score:.4f} at episode {best_episode}")
                break
        
        # Save models periodically
        if (episode + 1) % 10 == 0:
            multi_agent.save_models(f"{config.RESULT_DIR}/models_episode_{episode + 1}")
    
    # Save final models
    multi_agent.save_models(f"{config.RESULT_DIR}/final_models")
    
    # Plot training curves
    plot_training_curves(
        train_returns=train_returns,
        train_sharpe_ratios=train_sharpe_ratios,
        train_drawdowns=train_drawdowns,
        validation_returns=validation_returns,
        validation_sharpe_ratios=validation_sharpe_ratios,
        validation_drawdowns=validation_drawdowns
    )
    
    print(f"\n✅ Training completed! Best composite score: {best_composite_score:.4f} at episode {best_episode}")
    return best_composite_score, best_episode

def run_episode(multi_agent: MultiAgentSystem, data: pd.DataFrame, 
                returns_history: pd.DataFrame, train: bool = True, 
                batch_size: int = 64, populate_meta_buffer: bool = False) -> Dict[str, float]:
    """
    Run a single episode on the given data.
    
    Args:
        multi_agent: Multi-agent system to use
        data: Data to run the episode on
        returns_history: Historical returns data
        train: Whether to train the agents
        batch_size: Batch size for updates (0 for no updates)
        populate_meta_buffer: Whether to populate the meta-controller's replay buffer
        
    Returns:
        Dictionary of episode statistics
    """
    # Initialize portfolio
    balance = config.INITIAL_BALANCE
    holdings = np.zeros(config.NUM_STOCKS)
    portfolio_values = [balance]
    daily_returns = []
    
    # Extract price data and technical indicators
    prices_df = data[[col for col in data.columns if col.endswith('_Adj Close')]]
    technical_indicators = utils.calculate_technical_indicators(data)
    
    # Run through each day
    for t in range(1, len(data)):
        # Get current prices
        current_prices = np.array([prices_df.iloc[t-1, i] for i in range(config.NUM_STOCKS)])
        
        # Calculate current portfolio value
        portfolio_value = utils.calculate_portfolio_value(current_prices, holdings, balance)
        
        # Preprocess state
        state = utils.preprocess_state(
            prices=prices_df.iloc[t-1],
            holdings=holdings,
            balance=balance,
            technical_indicators=technical_indicators.iloc[t-1]
        )
        normalized_state = utils.normalize_state(state)
        
        # Populate meta-controller's replay buffer during training
        if train and populate_meta_buffer:
            multi_agent.add_experience_for_meta_controller(normalized_state)
        
        # Get appropriate slice of returns history
        current_slice_returns_history = returns_history.iloc[:t] if returns_history is not None and t > 0 and not returns_history.empty else None

        # Update agent weights periodically using meta-controller
        if t % config.META_UPDATE_FREQ == 0:
            multi_agent.update_agent_weights(normalized_state)

        # Select action
        action = multi_agent.select_action(
            state=normalized_state,
            explore=train,
            returns_history=current_slice_returns_history
        )
        
        # Validate actions to prevent excessive risk
        for i in range(len(action)):
            if action[i] != 0:  # Only process non-zero actions
                action[i] = utils.validate_action(
                    action=np.array([action[i]]),
                    holdings=np.array([holdings[i]]),
                    prices=np.array([current_prices[i]]),
                    balance=balance,
                    max_position_pct=config.MAX_POSITION_PCT
                )[0]  # Extract the single value from the array
        
        # Convert normalized action to actual trades
        trades = utils.denormalize_action(action)
        
        # Check if action is unsafe
        is_unsafe = multi_agent.is_action_unsafe(
            state=normalized_state,
            action=action,
            returns_history=current_slice_returns_history
        )
        
        # Execute trades
        log_dir = os.path.join(config.RESULT_DIR, "log")
        total_monetary_transaction_cost_for_step = 0
        executed_trade_values = np.zeros_like(trades, dtype=float)  # To store actual value of trades
        
        for i in range(config.NUM_STOCKS):
            # Skip if no trade
            if trades[i] == 0:
                continue
            
            # Calculate cost
            cost = trades[i] * current_prices[i]
            
            # Check if we have enough balance for buying
            if trades[i] > 0 and cost > balance:
                # Scale down the trade
                max_shares = int(balance / current_prices[i])
                trades[i] = max_shares
                cost = trades[i] * current_prices[i]
            
            # Before updating holdings
            if trades[i] < 0:
                # Limit sell to what we have (prevent short selling)
                trades[i] = max(trades[i], -holdings[i])
            
            # Execute trade
            if trades[i] != 0:
                trade_value = trades[i] * current_prices[i]
                executed_trade_values[i] = trade_value  # Store for reward calculation
                
                # Update holdings
                holdings[i] += trades[i]
                
                # Update balance
                balance -= trade_value
                
                # Calculate and accumulate transaction cost
                current_trade_transaction_cost = abs(trade_value) * config.TRANSACTION_COST
                total_monetary_transaction_cost_for_step += current_trade_transaction_cost
                
                # Apply transaction cost to balance
                balance -= current_trade_transaction_cost
                
                # Log the trade
                ticker = prices_df.columns[i].split('_')[0] if '_' in prices_df.columns[i] else prices_df.columns[i]
                timestamp = prices_df.index[t].strftime('%Y-%m-%d')
                portfolio_value = utils.calculate_portfolio_value(current_prices, holdings, balance)
                
                utils.log_trade(
                    log_dir=f"{config.RESULT_DIR}/log",
                    timestamp=timestamp,
                    ticker=ticker,
                    action=trades[i],
                    price=current_prices[i],
                    cost=abs(trade_value) * (1 + config.TRANSACTION_COST),
                    holdings_before=holdings[i] - trades[i],
                    holdings_after=holdings[i],
                    balance_before=balance + trade_value + current_trade_transaction_cost,
                    balance_after=balance,
                    portfolio_value=portfolio_value
                )
        
        # Get next prices
        next_prices = np.array([prices_df.iloc[t, i] for i in range(config.NUM_STOCKS)])
        
        # Calculate new portfolio value
        new_portfolio_value = utils.calculate_portfolio_value(next_prices, holdings, balance)

        # Emergency reset if portfolio value is negative
        if new_portfolio_value <= 0:
            print(f"CRITICAL LOSS @ {data.index[t]}: Portfolio value {new_portfolio_value:.2f}. Minimal recapitalization.")
            minimal_capital_floor = config.INITIAL_BALANCE * 0.01 # e.g., 1% of initial, or a fixed small amount like 1000
            balance = minimal_capital_floor
            holdings = np.zeros(config.NUM_STOCKS) # Liquidate all assets
            new_portfolio_value = balance # The new portfolio value is this recapitalized balance
            print(f"Portfolio recapitalized to {new_portfolio_value:.2f}")
            # Log this event if you have detailed event logging.
            # The reward for the step leading to this should still be very negative.

        portfolio_values.append(new_portfolio_value)
        
        # Calculate daily return
        daily_return = (new_portfolio_value - portfolio_value) / portfolio_value
        daily_returns.append(daily_return)
        
        # Log daily performance
        if len(portfolio_values) > 1:
            # Calculate rolling Sharpe ratio if we have enough data
            rolling_sharpe = 0
            if len(daily_returns) >= 30:
                rolling_returns = np.array(daily_returns[-30:])
                rolling_sharpe = utils.calculate_sharpe_ratio(rolling_returns)
            
            # Calculate max drawdown to date
            current_drawdown = utils.calculate_max_drawdown(np.array(portfolio_values))
            
            # Log the daily performance
            timestamp = prices_df.index[t].strftime('%Y-%m-%d')
            utils.log_daily_performance(
                log_dir=log_dir,
                timestamp=timestamp,
                portfolio_value=new_portfolio_value,
                previous_value=portfolio_value,
                daily_return=daily_return,
                sharpe_ratio=rolling_sharpe,
                max_drawdown=current_drawdown,
                holdings=holdings,
                prices=next_prices,
                balance=balance,
                agent_weights=multi_agent.agent_weights
            )
        
        # Calculate reward with improved risk adjustment
        if len(portfolio_values) > 1:
            return_component = (new_portfolio_value - portfolio_values[-2]) / portfolio_values[-2]
            
            # Calculate risk component
            if len(portfolio_values) > 30:
                # Calculate rolling volatility
                rolling_returns = np.diff(portfolio_values[-31:]) / portfolio_values[-31:-1]
                volatility = np.std(rolling_returns) * np.sqrt(252)
                
                # Calculate rolling drawdown
                rolling_values = portfolio_values[-31:]
                peak = np.maximum.accumulate(rolling_values)
                drawdown = (peak - rolling_values) / peak
                max_drawdown = np.max(drawdown)
                
                # Penalize high volatility and drawdowns
                risk_penalty = 0.5 * volatility + 2.0 * max_drawdown
            else:
                risk_penalty = 0
            
            # Use the accumulated actual monetary transaction cost for the reward penalty
            reward = return_component - total_monetary_transaction_cost_for_step - risk_penalty
            
            # Add safety penalty if action was unsafe
            if is_unsafe:
                reward -= config.SAFETY_WEIGHTS[0]  # Use most conservative penalty
        else:
            reward = 0
        
        # Get next state
        next_state = utils.preprocess_state(
            prices=prices_df.iloc[t],
            holdings=holdings,
            balance=balance,
            technical_indicators=technical_indicators.iloc[t]
        )
        normalized_next_state = utils.normalize_state(next_state)
        
        # Store experience in replay buffer if training
        if train:
            for agent in multi_agent.agents:
                agent.replay_buffer.add(
                    state=normalized_state,
                    action=action,
                    reward=reward,
                    next_state=normalized_next_state,
                    done=False,
                    unsafe=is_unsafe
                )
            
            # Update agents
            if batch_size > 0:
                losses = multi_agent.update_agents(batch_size, returns_history)
            
            # Update meta-controller weights periodically
            if t % config.META_UPDATE_FREQ == 0:
                multi_agent.update_agent_weights(normalized_state)
        
        # Fix any negative holdings (shouldn't happen, but just in case)
        for i in range(len(holdings)):
            if holdings[i] < 0:
                print(f"WARNING: Fixing negative holdings for stock {i}: {holdings[i]}")
                holdings[i] = 0
    
    # Calculate episode statistics
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    sharpe_ratio = utils.calculate_sharpe_ratio(np.array(daily_returns))
    max_drawdown = utils.calculate_max_drawdown(np.array(portfolio_values))
    
    return {
        "return": total_return,
        "sharpe": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "portfolio_values": portfolio_values,
        "daily_returns": daily_returns
    }

def test(multi_agent: MultiAgentSystem, test_data: pd.DataFrame, 
         returns_history: pd.DataFrame, baseline_data: pd.DataFrame = None):
    """
    Test the multi-agent system on the test data.
    
    Args:
        multi_agent: Multi-agent system to test
        test_data: Test data
        returns_history: Historical returns data
        baseline_data: Baseline data (e.g., DJI) for comparison
    """
    print("Starting testing...")
    
    # Run test episode
    test_stats = run_episode(
        multi_agent=multi_agent,
        data=test_data,
        returns_history=returns_history,
        train=False,
        batch_size=0
    )
    
    # Calculate composite performance score
    composite_score = utils.calculate_composite_performance_score(
        return_val=test_stats["return"],
        sharpe_ratio=test_stats["sharpe"],
        max_drawdown=test_stats["max_drawdown"]
    )
    
    # Print test statistics
    print(f"📈 Test Results:")
    print(f"  Return: {test_stats['return']:.4f}")
    print(f"  Sharpe Ratio: {test_stats['sharpe']:.4f}")
    print(f"  Max Drawdown: {test_stats['max_drawdown']:.4f}")
    print(f"  🎯 Composite Score: {composite_score:.4f}")
    
    # Save test statistics to CSV
    test_results_df = pd.DataFrame({
        'metric': ['return', 'sharpe_ratio', 'max_drawdown', 'composite_score'],
        'value': [test_stats['return'], test_stats['sharpe'], test_stats['max_drawdown'], composite_score]
    })
    test_results_df.to_csv(f"{config.RESULT_DIR}/test_results_metrics.csv", index=False)
    
    # Save detailed portfolio values and daily returns
    portfolio_df = pd.DataFrame({
        'date': test_data.index[:len(test_stats['portfolio_values'])],
        'portfolio_value': test_stats['portfolio_values'],
        'daily_return': [0] + test_stats['daily_returns']  # Add 0 for the first day
    })
    portfolio_df.to_csv(f"{config.RESULT_DIR}/test_results_detailed.csv", index=False)
    
    # Plot results
    plot_test_results(
        portfolio_values=test_stats["portfolio_values"],
        baseline_data=baseline_data,
        test_data=test_data
    )
    
    return composite_score

def plot_training_curves(train_returns: List[float], train_sharpe_ratios: List[float], 
                         train_drawdowns: List[float], validation_returns: List[float], 
                         validation_sharpe_ratios: List[float], validation_drawdowns: List[float]):
    """
    Plot training curves.
    
    Args:
        train_returns: List of training returns
        train_sharpe_ratios: List of training Sharpe ratios
        train_drawdowns: List of training max drawdowns
        validation_returns: List of validation returns
        validation_sharpe_ratios: List of validation Sharpe ratios
        validation_drawdowns: List of validation max drawdowns
    """
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot returns
    axes[0].plot(train_returns, label="Train")
    axes[0].plot(np.arange(0, len(train_returns), config.EVAL_FREQUENCY), validation_returns, label="Validation")
    axes[0].set_title("Returns")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].legend()
    
    # Plot Sharpe ratios
    axes[1].plot(train_sharpe_ratios, label="Train")
    axes[1].plot(np.arange(0, len(train_sharpe_ratios), config.EVAL_FREQUENCY), validation_sharpe_ratios, label="Validation")
    axes[1].set_title("Sharpe Ratios")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Sharpe Ratio")
    axes[1].legend()
    
    # Plot max drawdowns
    axes[2].plot(train_drawdowns, label="Train")
    axes[2].plot(np.arange(0, len(train_drawdowns), config.EVAL_FREQUENCY), validation_drawdowns, label="Validation")
    axes[2].set_title("Max Drawdowns")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Max Drawdown")
    axes[2].legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{config.RESULT_DIR}/training_curves.png")
    plt.close()

def plot_test_results(portfolio_values: List[float], baseline_data: pd.DataFrame, test_data: pd.DataFrame):
    """
    Plot test results.
    
    Args:
        portfolio_values: List of portfolio values
        baseline_data: Baseline data (e.g., DJI) for comparison
        test_data: Test data
    """
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Normalize portfolio values
    normalized_portfolio = np.array(portfolio_values) / portfolio_values[0]
    
    # Plot portfolio values
    plt.plot(normalized_portfolio, label="Multi-Agent Portfolio")
    
    # Plot baseline if available
    if baseline_data is not None:
        # Try to use Close column first, then fall back to other options
        if 'Close' in baseline_data.columns:
            adj_close_col = 'Close'
        else:
            # Find any column containing 'close'
            adj_close_col = None
            for col in baseline_data.columns:
                if 'close' in col.lower():
                    adj_close_col = col
                    break
            
            if adj_close_col is None and len(baseline_data.columns) > 0:
                # If no column with 'close' in name, use the first column
                adj_close_col = baseline_data.columns[0]
                print(f"Warning: No 'Close' column found in baseline data. Using '{adj_close_col}' instead.")
        
        try:
            # Extract baseline prices for the test period
            # Use reindex to handle index mismatches
            aligned_baseline = baseline_data[adj_close_col].reindex(test_data.index, method='ffill')
            normalized_baseline = aligned_baseline.values / aligned_baseline.values[0]
            plt.plot(normalized_baseline, label=f"Baseline (DJI - {adj_close_col})")
        except Exception as e:
            print(f"Warning: Error plotting baseline: {e}")
            print(f"Baseline columns: {baseline_data.columns.tolist()}")
    
    # Add labels and title
    plt.title("Portfolio Performance vs. Baseline")
    plt.xlabel("Trading Day")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True)
    
    # Save figure
    plt.savefig(f"{config.RESULT_DIR}/test_results.png")
    plt.close()

def main():
    """
    Main function.
    """
    # Create result and log directories
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.RESULT_DIR, "log"), exist_ok=True)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Multi-Agent Trading Strategy")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "both"],
                        help="Mode to run in (train, test, or both)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to load models from (for testing)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run on (cpu or cuda)")
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    
    # Load data
    print("Loading data...")
    train_data_df = utils.load_stock_data(config.TRAINING_DATA_PATH) # Renamed for clarity
    validation_data_df = utils.load_stock_data(config.VALIDATION_DATA_PATH) # Renamed
    test_data_df = utils.load_stock_data(config.TEST_DATA_PATH) # Renamed
    baseline_data = utils.load_stock_data(config.BASELINE_DATA_PATH)

    # Calculate returns history specific to each dataset
    # These will be the full histories for each dataset.
    # run_episode will then take slices like .iloc[:t]
    train_phase_returns_history = utils.calculate_returns(train_data_df[[col for col in train_data_df.columns if '_Adj Close' in col]])
    validation_phase_returns_history = utils.calculate_returns(validation_data_df[[col for col in validation_data_df.columns if '_Adj Close' in col]])
    test_phase_returns_history = utils.calculate_returns(test_data_df[[col for col in test_data_df.columns if '_Adj Close' in col]])
    # Ensure calculate_returns only uses Adj Close columns for this purpose.
    # If calculate_returns needs the full dataframe structure, pass train_data_df etc. directly.
    
    # Initialize multi-agent system
    state_dim = 1 + config.NUM_STOCKS * 6  # balance + (price, holdings, MACD, RSI, CCI, ADX) for each stock
    action_dim = config.NUM_STOCKS
    multi_agent = MultiAgentSystem(state_dim, action_dim, device=args.device)
    
    # Train or test
    if args.mode in ["train", "both"]:
        # Pass the appropriate returns_history for the training phase
        train_model(multi_agent, train_data_df, validation_data_df, 
                    train_phase_returns_history, validation_phase_returns_history) # Modified train call
        
        # --- START: CODE FIX ---
        # After training in 'both' mode, reload the best model to ensure the test uses
        # the best-performing checkpoint, not the final-epoch model.
        if args.mode == "both":
            best_model_path = f"{config.RESULT_DIR}/best_models"
            if os.path.exists(best_model_path):
                print(f"\nReloading best model from {best_model_path} for testing phase.")
                multi_agent.load_models(best_model_path)
            else:
                print("\nWarning: 'best_models' directory not found. Proceeding with final model from training.")
        # --- END: CODE FIX ---
    
    if args.mode in ["test", "both"]:
        # Load best models if available and not in "both" mode
        if args.mode == "test" and args.model_path is not None:
            multi_agent.load_models(args.model_path)
        elif args.mode == "test" and args.model_path is None:
            # Try to load best models from default location
            if os.path.exists(f"{config.RESULT_DIR}/best_models"):
                multi_agent.load_models(f"{config.RESULT_DIR}/best_models")
            else:
                print("Warning: No model path specified and no best models found. Using untrained models.")
        
        # Pass the appropriate returns_history for the test phase
        test_model(multi_agent, test_data_df, test_phase_returns_history, baseline_data) # Modified test call

def train_model(multi_agent: MultiAgentSystem, train_data: pd.DataFrame, 
          validation_data: pd.DataFrame, 
          train_returns_history: pd.DataFrame,
          validation_returns_history: pd.DataFrame):
    """
    Train the multi-agent system.
    
    Args:
        multi_agent: Multi-agent system to train
        train_data: Training data
        validation_data: Validation data
        train_returns_history: Historical returns data for training
        validation_returns_history: Historical returns data for validation
    """
    print("Starting training...")
    
    # Create result directory
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    
    # Training statistics
    train_returns = []
    train_sharpe_ratios = []
    train_drawdowns = []
    validation_returns = []
    validation_sharpe_ratios = []
    validation_drawdowns = []
    validation_composite_scores = []  # Track composite performance scores
    meta_controller_losses = []  # Track meta-controller losses
    
    # Early stopping variables
    best_composite_score = float('-inf')
    patience_counter = 0
    best_episode = 0
    
    # Training loop
    for episode in range(config.EPISODES):
        start_time = time.time()
        
        # Train on training data
        train_stats = run_episode(
            multi_agent=multi_agent,
            data=train_data,
            returns_history=train_returns_history,
            train=True,
            batch_size=config.BATCH_SIZE,
            populate_meta_buffer=True  # Populate meta-controller's buffer
        )
        
        train_returns.append(train_stats["return"])
        train_sharpe_ratios.append(train_stats["sharpe"])
        train_drawdowns.append(train_stats["max_drawdown"])
        
        # Periodically train the meta-controller
        if episode > 0 and (episode + 1) % config.META_TRAIN_FREQ == 0 and \
           len(multi_agent.meta_replay_buffer) >= multi_agent.meta_batch_size:
            meta_loss = multi_agent.train_meta_controller_from_buffer()
            meta_controller_losses.append(meta_loss)
            print(f"  Meta-Controller training: Episode {episode + 1}, Loss: {meta_loss:.4f}")
        
        # Evaluate on validation data
        if (episode + 1) % config.EVAL_FREQUENCY == 0:
            validation_stats = run_episode(
                multi_agent=multi_agent,
                data=validation_data,
                returns_history=validation_returns_history,
                train=False,
                batch_size=0,
                populate_meta_buffer=False  # Don't populate during validation
            )
            
            validation_returns.append(validation_stats["return"])
            validation_sharpe_ratios.append(validation_stats["sharpe"])
            validation_drawdowns.append(validation_stats["max_drawdown"])
            
            # Calculate composite performance score
            composite_score = utils.calculate_composite_performance_score(
                return_val=validation_stats["return"],
                sharpe_ratio=validation_stats["sharpe"],
                max_drawdown=validation_stats["max_drawdown"]
            )
            validation_composite_scores.append(composite_score)
            
            # Check for best performance and early stopping
            if composite_score > best_composite_score:
                best_composite_score = composite_score
                best_episode = episode + 1
                patience_counter = 0
                print(f"  🎯 New best composite score: {composite_score:.4f} (Return: {validation_stats['return']:.4f}, "
                      f"Sharpe: {validation_stats['sharpe']:.4f}, Max DD: {validation_stats['max_drawdown']:.4f})")
                print(f"  📁 Saving best models to {config.RESULT_DIR}/best_models")
                multi_agent.save_models(f"{config.RESULT_DIR}/best_models")
            else:
                patience_counter += config.EVAL_FREQUENCY
                print(f"  📊 Composite score: {composite_score:.4f} (No improvement for {patience_counter} episodes)")
            
            # Print statistics
            print(f"Episode {episode + 1}/{config.EPISODES} - Time: {time.time() - start_time:.2f}s")
            print(f"  Train - Return: {train_stats['return']:.4f}, Sharpe: {train_stats['sharpe']:.4f}, Max DD: {train_stats['max_drawdown']:.4f}")
            print(f"  Valid - Return: {validation_stats['return']:.4f}, Sharpe: {validation_stats['sharpe']:.4f}, Max DD: {validation_stats['max_drawdown']:.4f}")
            
            # Early stopping check
            if patience_counter >= config.PATIENCE:
                print(f"\n🛑 Early stopping triggered! No improvement for {config.PATIENCE} episodes.")
                print(f"Best composite score: {best_composite_score:.4f} at episode {best_episode}")
                break
        
        # Save models periodically
        if (episode + 1) % 10 == 0:
            multi_agent.save_models(f"{config.RESULT_DIR}/models_episode_{episode + 1}")
    
    # Save final models
    multi_agent.save_models(f"{config.RESULT_DIR}/final_models")
    
    # Save training statistics including composite scores
    training_stats_df = pd.DataFrame({
        'episode': range(config.EVAL_FREQUENCY, len(validation_composite_scores) * config.EVAL_FREQUENCY + 1, config.EVAL_FREQUENCY),
        'validation_return': validation_returns,
        'validation_sharpe': validation_sharpe_ratios,
        'validation_drawdown': validation_drawdowns,
        'validation_composite_score': validation_composite_scores
    })
    training_stats_df.to_csv(f"{config.RESULT_DIR}/training_validation_stats.csv", index=False)
    
    # Save meta-controller losses
    if meta_controller_losses:
        meta_loss_df = pd.DataFrame({
            'episode': range(config.META_TRAIN_FREQ, config.EPISODES + 1, config.META_TRAIN_FREQ)[:len(meta_controller_losses)],
            'loss': meta_controller_losses
        })
        meta_loss_df.to_csv(f"{config.RESULT_DIR}/meta_controller_losses.csv", index=False)
    
    # Plot training curves
    plot_training_curves(
        train_returns=train_returns,
        train_sharpe_ratios=train_sharpe_ratios,
        train_drawdowns=train_drawdowns,
        validation_returns=validation_returns,
        validation_sharpe_ratios=validation_sharpe_ratios,
        validation_drawdowns=validation_drawdowns
    )
    
    print(f"\n✅ Training completed! Best composite score: {best_composite_score:.4f} at episode {best_episode}")
    return best_composite_score, best_episode

def test_model(multi_agent: MultiAgentSystem, test_data: pd.DataFrame, 
         test_returns_history: pd.DataFrame,
         baseline_data: pd.DataFrame = None):
    """
    Test the multi-agent system on the test data.
    
    Args:
        multi_agent: Multi-agent system to test
        test_data: Test data
        test_returns_history: Historical returns data for testing
        baseline_data: Baseline data (e.g., DJI) for comparison
    """
    print("Starting testing...")
    
    # Run test episode
    test_stats = run_episode(
        multi_agent=multi_agent,
        data=test_data,
        returns_history=test_returns_history,
        train=False,
        batch_size=0,
        populate_meta_buffer=False  # Don't populate during testing
    )
    
    # Calculate composite performance score
    composite_score = utils.calculate_composite_performance_score(
        return_val=test_stats["return"],
        sharpe_ratio=test_stats["sharpe"],
        max_drawdown=test_stats["max_drawdown"]
    )
    
    # Print test statistics
    print(f"📈 Test Results:")
    print(f"  Return: {test_stats['return']:.4f}")
    print(f"  Sharpe Ratio: {test_stats['sharpe']:.4f}")
    print(f"  Max Drawdown: {test_stats['max_drawdown']:.4f}")
    print(f"  🎯 Composite Score: {composite_score:.4f}")
    
    # Save test statistics to CSV
    test_results_df = pd.DataFrame({
        'metric': ['return', 'sharpe_ratio', 'max_drawdown', 'composite_score'],
        'value': [test_stats['return'], test_stats['sharpe'], test_stats['max_drawdown'], composite_score]
    })
    test_results_df.to_csv(f"{config.RESULT_DIR}/test_results_metrics.csv", index=False)
    
    # Save detailed portfolio values and daily returns
    portfolio_df = pd.DataFrame({
        'date': test_data.index[:len(test_stats['portfolio_values'])],
        'portfolio_value': test_stats['portfolio_values'],
        'daily_return': [0] + test_stats['daily_returns']  # Add 0 for the first day
    })
    portfolio_df.to_csv(f"{config.RESULT_DIR}/test_results_detailed.csv", index=False)
    
    # Plot results
    plot_test_results(
        portfolio_values=test_stats["portfolio_values"],
        baseline_data=baseline_data,
        test_data=test_data
    )
    
    return composite_score

if __name__ == "__main__":
    main()