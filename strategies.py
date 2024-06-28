from abc import ABC, abstractmethod
from typing import Any
import pandas as pd

class Strategy(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals for the given data.
        
        :param data: DataFrame containing price and other relevant data
        :return: A Series with the same index as data, containing trading signals
        """
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        """
        Return a dictionary of the strategy's parameters.
        
        :return: A dictionary with parameter names as keys and their values
        """
        pass

    @abstractmethod
    def set_parameters(self, parameters: dict) -> None:
        """
        Set the strategy's parameters.
        
        :param parameters: A dictionary with parameter names as keys and their values
        """
        pass

    def __str__(self):
        return f"{self.name} Strategy"
    
class TrendCrossoverStrategy(Strategy):
    def __init__(self, fast_column='trend_ema_fast', slow_column='trend_ema_slow'):
        super().__init__("Trend Crossover")
        self.fast_column = fast_column
        self.slow_column = slow_column

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(index=data.index, data=0)
        
        crossover_up = (data[self.fast_column].shift(1) < data[self.slow_column].shift(1)) & \
                       (data[self.fast_column] > data[self.slow_column])
        crossover_down = (data[self.fast_column].shift(1) > data[self.slow_column].shift(1)) & \
                         (data[self.fast_column] < data[self.slow_column])
        
        signals[crossover_up] = 1
        signals[crossover_down] = -1
        
        return signals

    def get_parameters(self) -> dict:
        return {
            "fast_column": self.fast_column,
            "slow_column": self.slow_column
        }

    def set_parameters(self, parameters: dict) -> None:
        self.fast_column = parameters.get("fast_column", self.fast_column)
        self.slow_column = parameters.get("slow_column", self.slow_column)

class MLModelStrategy(Strategy):
    def __init__(self, name: str, model: Any, feature_columns: list):
        super().__init__(name)
        self.model = model
        self.feature_columns = feature_columns

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Ensure all feature columns are present in the data
        if not all(col in data.columns for col in self.feature_columns):
            missing_cols = set(self.feature_columns) - set(data.columns)
            raise ValueError(f"Missing columns in data: {missing_cols}")

        features = data[self.feature_columns]

        predictions = self.model.predict(features)

        signals = pd.Series(predictions, index=data.index)

        return signals

    def get_parameters(self) -> dict:
        return {
            "model": self.model,
            "feature_columns": self.feature_columns
        }

    def set_parameters(self, parameters: dict) -> None:
        self.model = parameters.get("model", self.model)
        self.feature_columns = parameters.get("feature_columns", self.feature_columns)

class Condition(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        pass

    def __str__(self):
        return f"{self.name} Condition"
    
class VolatilityCondition(Condition):
    def __init__(self, column='volatility_atr', threshold=1.75):
        super().__init__("Volatility")
        self.column = column
        self.threshold = threshold

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        return (data[self.column] > self.threshold).astype(int)