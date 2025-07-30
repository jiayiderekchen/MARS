"""
MARS: Multi-Agent Reinforcement Strategy
Utility functions for data processing, feature engineering, and risk management.
"""
import pandas as pd
import numpy as np
import os
import ta
from typing import List, Dict, Tuple, Optional
import config
import datetime

def load_stock_data(file_path: str) -> pd.DataFrame:
    """
    Load stock price data from CSV file.
    
    Args:
        file_path: Path to the CSV file containing stock data
        
    Returns:
        DataFrame with stock price data
    """
    # Try the path as is
    if os.path.exists(file_path):
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return data
    
    # Try with new_experiment_4 prefix
    alt_path = os.path.join("new_experiment_4", file_path)
    if os.path.exists(alt_path):
        data = pd.read_csv(alt_path, index_col=0, parse_dates=True)
        return data
    
    # If neither path works, raise an error
    raise FileNotFoundError(f"Could not find stock data at {file_path} or {alt_path}")

def load_stock_universe(file_path: str) -> List[str]:
    """
    Load the list of stocks in the universe.
    
    Args:
        file_path: Path to the CSV file containing stock tickers
        
    Returns:
        List of stock tickers
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Stock universe file not found: {file_path}")
    
    universe = pd.read_csv(file_path)
    return universe['Ticker'].tolist()

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for each stock.
    
    Args:
        data: DataFrame with stock price data
        
    Returns:
        DataFrame with added technical indicators
    """
    # Create a dictionary to store all new columns
    new_columns = {}
    
    # Process each stock
    for col in data.columns:
        if not col.endswith('_Adj Close'):
            continue
            
        ticker = col.split('_')[0]
        price_series_adj_close = data[col]
        
        # Attempt to get corresponding raw H, L, C series
        high_series = data.get(f"{ticker}_High", price_series_adj_close)
        low_series = data.get(f"{ticker}_Low", price_series_adj_close)
        close_series = data.get(f"{ticker}_Close", price_series_adj_close)
        
        # Calculate MACD
        macd = ta.trend.MACD(
            close=close_series, 
            window_fast=config.MACD_FAST, 
            window_slow=config.MACD_SLOW, 
            window_sign=config.MACD_SIGNAL
        )
        new_columns[f"{ticker}_MACD"] = macd.macd()
        
        # Calculate RSI
        rsi = ta.momentum.RSIIndicator(
            close=close_series, 
            window=config.RSI_PERIOD
        )
        new_columns[f"{ticker}_RSI"] = rsi.rsi()
        
        # Calculate CCI
        cci = ta.trend.CCIIndicator(
            high=high_series, 
            low=low_series,  
            close=close_series,
            window=config.CCI_PERIOD
        )
        new_columns[f"{ticker}_CCI"] = cci.cci()
        
        # Calculate ADX
        adx = ta.trend.ADXIndicator(
            high=high_series, 
            low=low_series,   
            close=close_series,
            window=config.ADX_PERIOD
        )
        new_columns[f"{ticker}_ADX"] = adx.adx()
    
    # Create a new DataFrame with all columns at once to avoid fragmentation
    indicators_df = pd.DataFrame(new_columns, index=data.index)
    
    # Combine with original data
    result = pd.concat([data, indicators_df], axis=1)
    
    # Fill NaN values that might be present in the indicators
    result = result.ffill().fillna(0)
    
    return result

def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns from price data.
    
    Args:
        prices: DataFrame with price data
        
    Returns:
        DataFrame with daily returns
    """
    return prices.pct_change().dropna()

def calculate_risk_metrics(returns: pd.DataFrame, window: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate rolling volatility and correlation matrix.
    
    Args:
        returns: DataFrame with daily returns
        window: Window size for rolling calculations
        
    Returns:
        Tuple of (volatility DataFrame, correlation DataFrame)
    """
    # Calculate rolling volatility
    volatility = returns.rolling(window=window).std() * np.sqrt(window)
    
    # Calculate rolling correlation matrix
    correlation = returns.rolling(window=window).corr()
    
    return volatility, correlation

def validate_action(action: np.ndarray, holdings: np.ndarray, prices: np.ndarray, 
                   balance: float, max_position_pct: float = 0.2) -> np.ndarray:
    """
    Apply risk management constraints to trading actions.
    
    Args:
        actions: Raw action array from the model
        holdings: Current stock holdings
        prices: Current stock prices
        balance: Current cash balance
        
    Returns:
        Adjusted action array
    """
    # Calculate current portfolio value
    total_stock_value = np.sum(holdings * prices)
    portfolio_value = balance + total_stock_value
    
    # If portfolio value is very small or negative, implement recovery mode
    if portfolio_value < 1000:
        print(f"RECOVERY MODE: Portfolio value {portfolio_value:.2f} is critically low")
        # Sell all negative-value positions
        for i in range(len(holdings)):
            if holdings[i] * prices[i] < 0:
                action[i] = -holdings[i]  # Liquidate this position
        return action
    
    # Calculate maximum position size (e.g., 20% of portfolio)
    max_position_value = portfolio_value * max_position_pct
    
    # Calculate current position values
    position_values = holdings * prices
    
    # Adjust actions to prevent excessive position sizes
    adjusted_action = action.copy()
    for i in range(len(action)):
        if action[i] > 0:  # Buying
            # Calculate how much we can buy
            current_value = position_values[i]
            max_additional_value = max(0, max_position_value - current_value)
            max_additional_shares = max_additional_value / prices[i] if prices[i] > 0 else 0
            
            # Limit buy action
            adjusted_action[i] = min(action[i], max_additional_shares)
        else:  # Selling
            # Can't sell more than we have
            adjusted_action[i] = max(action[i], -holdings[i])
    
    # Calculate the cost of the adjusted action
    action_cost = np.sum(adjusted_action * prices)
    
    # If action would result in negative balance, scale it down
    if balance - action_cost < 0:
        if action_cost > 0:  # Only scale down buys
            scale_factor = max(0, balance / action_cost)
            for i in range(len(adjusted_action)):
                if adjusted_action[i] > 0:  # Only scale down buys
                    adjusted_action[i] *= scale_factor
    
    return adjusted_action

def estimate_risk(state: np.ndarray, action: np.ndarray, returns_history: pd.DataFrame) -> float:
    """
    Improved risk estimation with more sophisticated metrics.
    """
    # Extract current holdings from state
    holdings = []
    for i in range(config.NUM_STOCKS):
        idx = 1 + i * 6 + 1  # Skip balance, then skip price for each stock
        holdings.append(state[idx])
    
    holdings = np.array(holdings)
    
    # Calculate new holdings after action
    new_holdings = holdings + action
    
    # Calculate portfolio concentration (Herfindahl-Hirschman Index)
    if np.sum(new_holdings) > 0:
        portfolio_weights = np.abs(new_holdings) / np.sum(np.abs(new_holdings))
        concentration_risk = np.sum(portfolio_weights ** 2)
    else:
        concentration_risk = 0
    
    # Calculate leverage risk (improved formula)
    leverage = np.sum(np.abs(new_holdings)) / (np.sum(new_holdings) + 1e-10)
    leverage_risk = min(leverage / 2.0, 1.0)  # More conservative cap
    
    # Calculate volatility risk using historical returns
    volatility_risk = 0
    if returns_history is not None and len(returns_history) > 0:
        # Get the most recent 30 days of returns
        recent_returns = returns_history.iloc[-30:].values
        
        if np.sum(np.abs(new_holdings)) > 0:
            portfolio_weights = np.abs(new_holdings) / np.sum(np.abs(new_holdings))
            
            # Calculate portfolio returns
            portfolio_returns = np.dot(recent_returns, portfolio_weights)
            
            # Calculate volatility (annualized)
            volatility = np.std(portfolio_returns) * np.sqrt(252)
            volatility_risk = min(volatility / 0.3, 1.0)  # More conservative cap
            
            # Add drawdown risk based on historical simulation
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (peak - cumulative_returns) / peak
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            drawdown_risk = min(max_drawdown / 0.1, 1.0)  # Cap at 10% drawdown
            
            # Add correlation risk (penalize high correlation with market)
            if 'market_return' in returns_history.columns:
                market_returns = returns_history['market_return'].iloc[-30:].values
                correlation = np.corrcoef(portfolio_returns, market_returns)[0, 1]
                correlation_risk = max(0, correlation)  # Only penalize positive correlation
            else:
                correlation_risk = 0
                
            # Combine all risk factors
            volatility_risk = 0.4 * volatility_risk + 0.4 * drawdown_risk + 0.2 * correlation_risk
    
    # Combine risk factors with adjusted weights
    total_risk = 0.3 * concentration_risk + 0.3 * leverage_risk + 0.4 * volatility_risk
    
    return min(total_risk, 1.0)

def preprocess_state(prices: pd.Series, holdings: np.ndarray, balance: float, 
                    technical_indicators: pd.Series) -> np.ndarray:
    """
    Preprocess the raw state into a format suitable for the agent.
    
    Args:
        prices: Current prices for all stocks
        holdings: Current holdings for all stocks
        balance: Current cash balance
        technical_indicators: Technical indicators for all stocks
        
    Returns:
        Preprocessed state vector
    """
    # Start with balance
    state = [balance]
    
    # Add price and holdings for each stock
    for i, ticker in enumerate(prices.index):
        # Extract ticker from column name if needed
        if '_Adj Close' in ticker:
            ticker_base = ticker.split('_')[0]
        else:
            ticker_base = ticker
            
        # Add price
        state.append(prices[ticker])
        
        # Add holdings
        state.append(holdings[i])
        
        # Add technical indicators - handle different possible structures
        try:
            # Try direct column access first
            macd_col = f"{ticker_base}_MACD"
            rsi_col = f"{ticker_base}_RSI"
            cci_col = f"{ticker_base}_CCI"
            adx_col = f"{ticker_base}_ADX"
            
            if macd_col in technical_indicators:
                state.append(technical_indicators[macd_col])
            else:
                state.append(0)  # Default MACD
                
            if rsi_col in technical_indicators:
                state.append(technical_indicators[rsi_col])
            else:
                state.append(50)  # Default RSI
                
            if cci_col in technical_indicators:
                state.append(technical_indicators[cci_col])
            else:
                state.append(0)  # Default CCI
                
            if adx_col in technical_indicators:
                state.append(technical_indicators[adx_col])
            else:
                state.append(25)  # Default ADX
                
        except Exception as e:
            # Fallback to default values if any error occurs
            print(f"Warning: Error accessing technical indicators for {ticker_base}: {e}")
            state.append(0)    # Default MACD
            state.append(50)   # Default RSI
            state.append(0)    # Default CCI
            state.append(25)   # Default ADX
    
    return np.array(state)

def normalize_state(state: np.ndarray) -> np.ndarray:
    """
    Normalize the state vector for neural network input.
    
    Args:
        state: Raw state vector
        
    Returns:
        Normalized state vector
    """
    # Extract components
    balance = state[0]
    
    normalized_state = [balance / 1000000.0]  # Normalize balance to millions
    
    for i in range(config.NUM_STOCKS):
        base_idx = 1 + i * 6
        
        # Price (normalize to range [0, 1] assuming max price of $10,000)
        price = state[base_idx]
        normalized_state.append(min(price / 10000.0, 1.0))
        
        # Holdings (normalize to range [-1, 1] assuming max holdings of 1000)
        holdings = state[base_idx + 1]
        normalized_state.append(max(min(holdings / 1000.0, 1.0), -1.0))
        
        # MACD (typically in range [-10, 10])
        macd = state[base_idx + 2]
        normalized_state.append(max(min(macd / 10.0, 1.0), -1.0))
        
        # RSI (already in range [0, 100])
        rsi = state[base_idx + 3]
        normalized_state.append(rsi / 100.0)
        
        # CCI (typically in range [-300, 300])
        cci = state[base_idx + 4]
        normalized_state.append(max(min(cci / 300.0, 1.0), -1.0))
        
        # ADX (already in range [0, 100])
        adx = state[base_idx + 5]
        normalized_state.append(adx / 100.0)
    
    return np.array(normalized_state)

def denormalize_action(action: np.ndarray) -> np.ndarray:
    """
    Convert normalized actions to actual trade sizes.
    
    Args:
        action: Normalized action vector in range [-1, 1]
        
    Returns:
        Trade sizes as integer values
    """
    # Scale to max trade size and round to integers
    trades = np.round(action * config.MAX_TRADE_SIZE).astype(int)
    return trades

def calculate_portfolio_value(prices: np.ndarray, holdings: np.ndarray, balance: float) -> float:
    """
    Calculate the total portfolio value.
    
    Args:
        prices: Current prices for all stocks
        holdings: Current holdings for all stocks
        balance: Current cash balance
        
    Returns:
        Total portfolio value
    """
    stock_value = np.sum(prices * holdings)
    return balance + stock_value

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe ratio of a return series.
    
    Args:
        returns: Array of daily returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
        
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    
    if np.std(excess_returns) == 0:
        return 0.0
        
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    annualized_sharpe = sharpe * np.sqrt(252)
    
    return annualized_sharpe

def calculate_max_drawdown(values: np.ndarray) -> float:
    """
    Calculate the maximum drawdown of a value series.
    
    Args:
        values: Array of portfolio values
        
    Returns:
        Maximum drawdown as a positive percentage
    """
    if len(values) < 2:
        return 0.0
        
    # Calculate the running maximum
    running_max = np.maximum.accumulate(values)
    
    # Calculate the drawdown
    drawdown = (running_max - values) / running_max
    
    # Return the maximum drawdown
    return np.max(drawdown)

def estimate_risk_batch(states: np.ndarray, actions: np.ndarray, returns_history: Optional[pd.DataFrame] = None) -> np.ndarray:
    """
    Estimate risk for a batch of state-action pairs.
    
    Args:
        states: Batch of states
        actions: Batch of actions
        returns_history: Historical returns data for risk estimation
        
    Returns:
        Batch of risk scores between 0 and 1
    """
    batch_size = states.shape[0]
    risks = np.zeros(batch_size)
    
    # Process each state-action pair individually
    for i in range(batch_size):
        risks[i] = estimate_risk(states[i], actions[i], returns_history)
    
    return risks

def log_trade(log_dir: str, timestamp: str, ticker: str, action: float, 
              price: float, cost: float, holdings_before: float, holdings_after: float,
              balance_before: float, balance_after: float, portfolio_value: float):
    """
    Log individual trades to a CSV file.
    
    Args:
        log_dir: Directory to save the log
        timestamp: Date of the trade
        ticker: Stock ticker
        action: Number of shares traded (positive for buy, negative for sell)
        price: Price per share
        cost: Total cost of the trade including transaction costs
        holdings_before: Holdings before the trade
        holdings_after: Holdings after the trade
        balance_before: Cash balance before the trade
        balance_after: Cash balance after the trade
        portfolio_value: Portfolio value after the trade
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file if it doesn't exist
    log_file = os.path.join(log_dir, "trade_log.csv")
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, 'a') as f:
        # Write header if file doesn't exist
        if not file_exists:
            f.write("date,ticker,action,price,cost,holdings_before,holdings_after,balance_before,balance_after,portfolio_value\n")
        
        # Write trade data
        f.write(f"{timestamp},{ticker},{action:.2f},{price:.2f},{cost:.2f},{holdings_before:.2f},{holdings_after:.2f},{balance_before:.2f},{balance_after:.2f},{portfolio_value:.2f}\n")

def log_daily_performance(log_dir: str, timestamp: str, portfolio_value: float, 
                         previous_value: float, daily_return: float, 
                         sharpe_ratio: float, max_drawdown: float,
                         holdings: np.ndarray, prices: np.ndarray,
                         balance: float, agent_weights: Optional[np.ndarray] = None):
    """
    Log daily portfolio performance to a CSV file.
    
    Args:
        log_dir: Directory to save the log
        timestamp: Date of the performance record
        portfolio_value: Current portfolio value
        previous_value: Previous day's portfolio value
        daily_return: Daily return percentage
        sharpe_ratio: Rolling Sharpe ratio
        max_drawdown: Maximum drawdown to date
        holdings: Current holdings of each stock
        prices: Current prices of each stock
        balance: Current cash balance
        agent_weights: Current weights of the agents
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file if it doesn't exist
    log_file = os.path.join(log_dir, "daily_pnl.csv")
    file_exists = os.path.isfile(log_file)
    
    # Recalculate portfolio value to ensure consistency
    total_stock_value = np.sum(holdings * prices)
    recalculated_portfolio_value = balance + total_stock_value
    
    # Safety check - ensure portfolio value is consistent
    if abs(recalculated_portfolio_value - portfolio_value) > 1.0:
        print(f"Warning: Portfolio value inconsistency detected on {timestamp}")
        print(f"  Provided: {portfolio_value:.2f}, Recalculated: {recalculated_portfolio_value:.2f}")
        portfolio_value = recalculated_portfolio_value
    
    # Calculate daily PnL
    daily_pnl = portfolio_value - previous_value
    
    # Recalculate daily return
    if previous_value > 0:
        daily_return = daily_pnl / previous_value
    else:
        daily_return = 0.0
    
    # Safety check - ensure portfolio value is not negative
    if portfolio_value < 0:
        print(f"Warning: Negative portfolio value detected on {timestamp}: {portfolio_value:.2f}")
        print(f"  Cash balance: {balance:.2f}, Stock value: {total_stock_value:.2f}")
        # For logging purposes, we'll still record the negative value to track the issue
    
    # Calculate allocation percentages
    if portfolio_value > 0:
        cash_allocation = balance / portfolio_value
        stock_allocation = total_stock_value / portfolio_value
    else:
        # If portfolio value is negative or zero, set allocations to 0
        cash_allocation = 0.0
        stock_allocation = 0.0
        print(f"Warning: Zero or negative portfolio value on {timestamp}, setting allocations to 0")
    
    with open(log_file, 'a') as f:
        # Write header if file doesn't exist
        if not file_exists:
            header = "date,portfolio_value,daily_pnl,daily_return,sharpe_ratio,max_drawdown,cash_allocation,stock_allocation"
            if agent_weights is not None:
                for i in range(len(agent_weights)):
                    header += f",agent{i}_weight"
            f.write(header + "\n")
        
        # Write performance data
        line = f"{timestamp},{portfolio_value:.2f},{daily_pnl:.2f},{daily_return:.6f},{sharpe_ratio:.4f},{max_drawdown:.4f},{cash_allocation:.4f},{stock_allocation:.4f}"
        if agent_weights is not None:
            for weight in agent_weights:
                line += f",{weight:.4f}"
        f.write(line + "\n")

def calculate_composite_performance_score(return_val: float, sharpe_ratio: float, max_drawdown: float, 
                                        weights: Dict[str, float] = None) -> float:
    """
    Calculate a composite performance score combining return, Sharpe ratio, and drawdown.
    
    Args:
        return_val: Portfolio return
        sharpe_ratio: Sharpe ratio
        max_drawdown: Maximum drawdown (positive value)
        weights: Dictionary with weights for 'return', 'sharpe', and 'drawdown'
    
    Returns:
        Composite performance score (higher is better)
    """
    if weights is None:
        import config
        weights = config.PERFORMANCE_WEIGHTS
    
    # Normalize metrics to comparable scales
    # Return: already in reasonable range (-1 to 5+)
    normalized_return = return_val
    
    # Sharpe: clip to reasonable range and normalize
    normalized_sharpe = np.clip(sharpe_ratio, -3, 5) / 5.0
    
    # Drawdown: convert to positive score (lower drawdown = higher score)
    # Clip drawdown to reasonable range (0 to 1) and invert
    normalized_drawdown_penalty = -np.clip(max_drawdown, 0, 1)  # Negative because drawdown is bad
    
    # Calculate weighted composite score
    composite_score = (
        weights['return'] * normalized_return +
        weights['sharpe'] * normalized_sharpe +
        weights['drawdown'] * normalized_drawdown_penalty
    )
    
    return composite_score 