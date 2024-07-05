import pandas as pd
import numpy as np
import ta  
from pathlib import Path
from sklearn.model_selection import train_test_split
import os
from pandas.errors import EmptyDataError
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

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

def get_features_df(stock_df: pd.DataFrame, tickers: List[str], start_date: str, end_date: str, mode: str, task: str = 'classification') -> pd.DataFrame:
    """
    Generate a DataFrame with technical indicators and target.

    :param stock_df: Input DataFrame containing stock data
    :param tickers: List of stock tickers to process
    :param start_date: Start date for filtering data
    :param end_date: End date for filtering data
    :param mode: 'train' or 'backtest' mode
    :param task: 'classification' or 'regression' task
    :return: DataFrame with added technical indicators and target.
    """
    if mode == 'train' and Path(f'train_df_{task}.csv').is_file():
        df = pd.read_csv(f'train_df_{task}.csv')
        df.drop(columns=['Unnamed: 0'], inplace=True)
        return df

    if mode == 'backtest' and Path(f'backtest_df_{task}.csv').is_file():
        df = pd.read_csv(f'backtest_df_{task}.csv')
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
        Process a single ticker's DataFrame to add technical indicators and target.

        :param ticker_df: DataFrame for a single ticker
        :return: Processed DataFrame with added features and target
        """
        if len(ticker_df) > 100:
            ticker_df = ta.add_all_ta_features(
                ticker_df, open="Open", high="High", low="Low", close="Close", volume="Volume"
            )
            
            if task == 'classification':
                ticker_df['target'] = np.where(ticker_df['Open'] <= ticker_df['Close'], 1, -1)
            elif task == 'regression':
                ticker_df['target'] = ticker_df['Close'].shift(-1)
            else:
                raise ValueError("Task must be either 'classification' or 'regression'")
            
            ticker_df.drop(['trend_psar_up', 'trend_psar_down'], axis=1, inplace=True)
            ticker_df.dropna(inplace=True)
            return ticker_df.iloc[:-1]
        
        return pd.DataFrame()
    
    processed_df = filtered_df.groupby('Ticker').apply(process_ticker).reset_index(drop=True)
    if mode == 'train':
        processed_df.to_csv(f'train_df_{task}.csv')
    elif mode == 'backtest':
        processed_df.to_csv(f'backtest_df_{task}.csv')

    return processed_df

def get_rnn_dataset(df, seq_length):
    df = df.sort_values(['Ticker', 'Date'])
    feature_columns = [col for col in df.columns if col not in ['Date', 'Ticker', 'Unnamed: 0', 'target']]
    grouped = df.groupby('Ticker')

    def create_sequences(group, seq_length):
        sequences = []
        targets = []
        
        for i in range(len(group) - seq_length - 1):  # -1 to avoid lookahead bias
            seq = group.iloc[i:i+seq_length]
            target = group.iloc[i+seq_length]['Close']  # Next day's close
            
            sequences.append(seq[feature_columns].values)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)

    X, y = zip(*grouped.apply(lambda x: create_sequences(x, seq_length)))

    # Concatenate all sequences and targets
    X = np.concatenate(X)
    y = np.concatenate(y)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    class StockDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    dataset = StockDataset(X_tensor, y_tensor)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset, feature_columns

def get_train_test(stock_df: pd.DataFrame, tickers: List[str], start_date: str, end_date: str, task: str = 'classification') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.

    :param stock_df: Input DataFrame containing stock data
    :param tickers: List of stock tickers to process
    :param start_date: Start date for filtering data
    :param end_date: End date for filtering data
    :param signal_threshold: Threshold for generating trading signals (optional)
    :return: Tuple containing X_train, X_test, y_train, y_test
    """
    train_df = get_features_df(stock_df, tickers, start_date, end_date, 'train', task)

    excluded_features = ['Date', 'Ticker']
    features = train_df.drop(columns=excluded_features + ['target'])
    target = train_df['target']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def get_features(df: pd.DataFrame) -> List[str]:
    """
    Get the list of feature columns from the DataFrame.

    :param df: Input DataFrame
    :return: List of feature column names
    """
    return list(df.drop(columns=['Date', 'Ticker', 'target']).columns)