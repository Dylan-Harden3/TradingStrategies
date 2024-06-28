from typing import List, Tuple
import numpy as np
import pandas as pd
from strategies import Strategy, MLModelStrategy, Condition
from scipy import stats

def gen_signals(df: pd.DataFrame, date: str, main_strategy: MLModelStrategy, conditions: List[Condition], additional_strategies: List[Strategy] = []) -> pd.DataFrame:
    """
    Generate trading signals for a specific date using the main strategy, conditions, and additional strategies.

    :param df: DataFrame containing all trading data
    :param date: The date for which to generate signals
    :param main_strategy: The primary ML strategy to use
    :param conditions: List of conditions to apply to the signals
    :param additional_strategies: List of additional strategies to apply
    :return: DataFrame with generated signals for the specified date
    """
    date_df = df.loc[df['Date'] == date].copy()
    if 'Unnamed: 0' in list(date_df.columns):
        date_df.drop(columns=['Unnamed: 0'], inplace=True)
    
    # Generate signals using the main strategy (ML model)
    date_df['Signal'] = main_strategy.generate_signals(date_df)
    
    # Apply all conditions
    for condition in conditions:
        condition_result = condition.evaluate(date_df)
        date_df['Signal'] *= condition_result
    
    # Apply additional strategies
    for strategy in additional_strategies:
        strategy_signals = strategy.generate_signals(date_df)
        date_df['Signal'] = np.where(strategy_signals == date_df['Signal'], date_df['Signal'], 0)
    
    return date_df[date_df['Signal'] != 0]

def gen_trades(df: pd.DataFrame, main_strategy: MLModelStrategy, conditions: List[Condition], additional_strategies: List[Strategy] = [], disable_shorting=False) -> pd.DataFrame:
    """
    Generate trades for all dates in the DataFrame.

    :param df: DataFrame containing all trading data
    :param main_strategy: The primary ML strategy to use
    :param conditions: List of conditions to apply to the signals
    :param additional_strategies: List of additional strategies to apply
    :return: DataFrame with generated trades for all dates
    """
    dataframes = []
    days = sorted(df['Date'].unique())

    # iterate through all the trading days
    for data_day, pred_day in zip(days, days[1:]):
        signals = gen_signals(df, data_day, main_strategy, conditions, additional_strategies)

        pred_day_data = df[df['Date'] == pred_day].set_index('Ticker')

        signals['Buy'] = np.nan
        signals['Sell'] = np.nan

        buy_mask = signals['Signal'] == 1
        sell_mask = signals['Signal'] == -1

        if not pred_day_data.empty:
            valid_buy_tickers = signals.loc[buy_mask, 'Ticker'].isin(pred_day_data.index)
            signals.loc[buy_mask & valid_buy_tickers, 'Buy'] = pred_day_data.loc[signals.loc[buy_mask & valid_buy_tickers, 'Ticker'], 'Open'].values
            signals.loc[buy_mask & valid_buy_tickers, 'Sell'] = pred_day_data.loc[signals.loc[buy_mask & valid_buy_tickers, 'Ticker'], 'Close'].values

            if not disable_shorting:
                valid_sell_tickers = signals.loc[sell_mask, 'Ticker'].isin(pred_day_data.index)
                signals.loc[sell_mask & valid_sell_tickers, 'Buy'] = pred_day_data.loc[signals.loc[sell_mask & valid_sell_tickers, 'Ticker'], 'Close'].values
                signals.loc[sell_mask & valid_sell_tickers, 'Sell'] = pred_day_data.loc[signals.loc[sell_mask & valid_sell_tickers, 'Ticker'], 'Open'].values

        dataframes.append(signals)
    
    trades_df = pd.concat(dataframes, ignore_index=True)
    trades_df['Date'] = pd.to_datetime(trades_df['Date'], format='%Y-%m-%d')
    return trades_df

def compute_daily_returns(trades_df: pd.DataFrame, initial_capital: float = 100000) -> Tuple[List[pd.Timestamp], List[float], List[float], List[float], int, int, int, int, int]:
    """
    Compute daily returns and other metrics based on the trades.

    :param trades_df: DataFrame containing trade information
    :param initial_capital: Initial capital for the portfolio
    :return: Tuple containing lists of dates, daily returns, cumulative returns, portfolio values,
             and integers for number of trades, longs, shorts, long wins, and short wins
    """
    dates = []
    daily_returns = []
    cumulative_returns = []
    portfolio_values = []

    capital = initial_capital
    portfolio_value = capital
    cum_return = 1

    num_trades = 0
    num_longs = 0
    num_shorts = 0
    num_long_wins = 0
    num_short_wins = 0

    for date in sorted(trades_df['Date'].unique()):
        daily_trades = trades_df[trades_df['Date'] == date]
        daily_return = 0

        if not daily_trades.empty:
            for _, trade in daily_trades.iterrows():
                if trade['Signal'] != 0:
                    buy_price = trade['Buy']
                    sell_price = trade['Sell']
                    if pd.notna(buy_price) and pd.notna(sell_price):
                        profit_loss = sell_price - buy_price
                        if trade['Signal'] == 1:
                            num_longs += 1
                        elif trade['Signal'] == -1:
                            num_shorts += 1

                        if profit_loss > 0:
                            if trade['Signal'] == 1:
                                num_long_wins += 1
                            elif trade['Signal'] == -1:
                                num_short_wins += 1
                        num_trades += 1
                        daily_return += profit_loss / portfolio_value
                        portfolio_value += profit_loss
        
        cum_return *= (1 + daily_return)
        capital = portfolio_value
        
        dates.append(date)
        daily_returns.append(daily_return)
        cumulative_returns.append(cum_return)
        portfolio_values.append(portfolio_value)
    
    return dates, daily_returns, cumulative_returns, portfolio_values, num_trades, num_longs, num_shorts, num_long_wins, num_short_wins

def compute_metrics(dates: List[pd.Timestamp], daily_returns: List[float], cumulative_returns: List[float], num_trades: int, num_longs: int, num_shorts: int, num_long_wins: int, num_short_wins: int) -> dict:
    """
    Compute various performance metrics based on the trading results.

    :param dates: List of dates for the trading period
    :param daily_returns: List of daily returns
    :param cumulative_returns: List of cumulative returns
    :param num_trades: Total number of trades
    :param num_longs: Number of long trades
    :param num_shorts: Number of short trades
    :param num_long_wins: Number of winning long trades
    :param num_short_wins: Number of winning short trades
    :return: Dictionary containing computed metrics
    """
    daily_returns = pd.Series(daily_returns, index=dates)
    cumulative_returns = pd.Series(cumulative_returns, index=dates)

    total_return = cumulative_returns.iloc[-1] - 1
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    
    best_day = daily_returns.max()
    worst_day = daily_returns.min()
    
    win_rate = (num_long_wins + num_short_wins) / num_trades if num_trades > 0 else 0
    long_win_rate = num_long_wins / num_longs if num_longs > 0 else 0
    short_win_rate = num_short_wins / num_shorts if num_shorts > 0 else 0
    
    metrics = {
        'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Best Day': best_day,
        'Worst Day': worst_day,
        'Number of Trades': num_trades,
        'Win Rate': win_rate,
        'Long Win Rate': long_win_rate,
        'Short Win Rate': short_win_rate,
    }

    return metrics