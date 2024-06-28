import pandas as pd
import numpy as np
import ta  
from pathlib import Path
from sklearn.model_selection import train_test_split
import os
from pandas.errors import EmptyDataError
from typing import List, Tuple

def load_df(tickers: List[str]) -> pd.DataFrame:
    """
    Load stock data from text files for specified tickers.

    :param tickers: List of stock tickers to load
    :return: DataFrame containing combined stock data for all tickers
    """
    dataframes = []
    directory = os.path.join(os.getcwd() + '/Stocks')

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            ticker = filename.replace('.us.txt', '').upper()
            file_path = os.path.join(directory, filename)

            if ticker not in tickers:  # if the ticker isn't in our list, ignore it
                continue

            try:
                df = pd.read_csv(file_path)
                df['Ticker'] = ticker
                df.sort_values('Date', inplace=True)
                dataframes.append(df)
            except EmptyDataError:  # some of the files have no contents so read_csv throws error
                pass

    stock_df = pd.concat(dataframes, ignore_index=True)

    stock_df['Date'] = pd.to_datetime(stock_df['Date'], format='%Y-%m-%d')
    if 'Unnamed: 0' in list(stock_df.columns):
        stock_df.drop(columns=['Unnamed: 0'], inplace=True)

    return stock_df

def get_features_df(stock_df: pd.DataFrame, tickers: List[str], start_date: str, end_date: str, mode: str, signal_threshold: float = None) -> pd.DataFrame:
    """
    Generate a DataFrame with technical indicators and trading signals.

    :param stock_df: Input DataFrame containing stock data
    :param tickers: List of stock tickers to process
    :param start_date: Start date for filtering data
    :param end_date: End date for filtering data
    :param mode: 'train' or 'backtest' mode
    :param signal_threshold: Threshold for generating trading signals (optional)
    :return: DataFrame with added technical indicators and trading signals
    """
    if mode == 'train' and Path(f'train_df_{signal_threshold}.csv').is_file():
        df = pd.read_csv(f'train_df_{signal_threshold}.csv')
        df.drop(columns=['Unnamed: 0'], inplace=True)
        return df

    if mode == 'backtest' and Path(f'backtest_df.csv').is_file():
        df = pd.read_csv(f'backtest_df.csv')
        df.drop(columns=['Unnamed: 0'], inplace=True)
        return df
    
    if 'Unnamed: 0' in list(stock_df.columns):
        stock_df.drop(columns=['Unnamed: 0'], inplace=True)
    
    filtered_df = stock_df.loc[
        (stock_df['Ticker'].isin(tickers)) & 
        (stock_df['Date'] >= start_date) & 
        (stock_df['Date'] <= end_date)
    ].copy()

    def process_ticker(ticker_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single ticker's DataFrame to add technical indicators and trading signals.

        :param ticker_df: DataFrame for a single ticker
        :return: Processed DataFrame with added features and signals
        """
        if len(ticker_df) > 100:
            ticker_df = ta.add_all_ta_features(
                ticker_df, open="Open", high="High", low="Low", close="Close", volume="Volume"
            )
            if signal_threshold:
                ticker_df['trading_signal'] = np.where(((ticker_df['Close'] - ticker_df['Open']) / ticker_df['Open']) >= signal_threshold, 1, -1)
            else:
                ticker_df['trading_signal'] = np.where(ticker_df['Close'] >= ticker_df['Open'], 1, -1)
            ticker_df.drop(['trend_psar_up', 'trend_psar_down'], axis=1, inplace=True)
            ticker_df.dropna(inplace=True)
            ticker_df['trading_signal'] = ticker_df['trading_signal'].shift(-1)
            return ticker_df.iloc[:-1]
        
        return pd.DataFrame()
    
    processed_df = filtered_df.groupby('Ticker').apply(process_ticker).reset_index(drop=True)
    if mode == 'train':
        processed_df.to_csv(f'train_df_{signal_threshold}.csv')
    elif mode == 'backtest':
        processed_df.to_csv(f'backtest_df.csv')

    return processed_df

def get_train_test(stock_df: pd.DataFrame, tickers: List[str], start_date: str, end_date: str, signal_threshold: float = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.

    :param stock_df: Input DataFrame containing stock data
    :param tickers: List of stock tickers to process
    :param start_date: Start date for filtering data
    :param end_date: End date for filtering data
    :param signal_threshold: Threshold for generating trading signals (optional)
    :return: Tuple containing X_train, X_test, y_train, y_test
    """
    train_df = get_features_df(stock_df, tickers, start_date, end_date, 'train', signal_threshold)

    excluded_features = ['Date', 'Ticker']
    features = train_df.drop(columns=excluded_features + ['trading_signal'])
    target = train_df['trading_signal']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def get_features(df: pd.DataFrame) -> List[str]:
    """
    Get the list of feature columns from the DataFrame.

    :param df: Input DataFrame
    :return: List of feature column names
    """
    return list(df.drop(columns=['Date', 'Ticker', 'trading_signal']).columns)