import pandas as pd
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import torch 
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

stocks = ["btc-usd", "eth-usd", "sol-usd", "aapl", "msft", "nvda"]

class Stock(Dataset):
    def __init__(self, stocks=stocks, window=128, train=True, test_year=2024):
        """
        Args:
            stocks (list): List of stock tickers to include.
            window (int): Size of the context window.
            train (bool): Whether to use training or testing data.
            min_year (int): Minimum year to include in the data.
            test_year (int): Year from which test data starts.
        """
        self.window = window
        self.train = train
        self.stocks = stocks

        # Dictionary to store normalized data for each stock
        normalized_data = {}
        common_dates = None  # To find the intersection of all dates

        # Process each stock
        for stock in self.stocks:
            tick = yf.Ticker(stock)
            history = tick.history(period="max")['Close']
            history.index = pd.to_datetime(history.index.date)

            # Normalize the data
            scaler = StandardScaler()
            norm_data = scaler.fit_transform(history.to_numpy().reshape(-1, 1))
            norm_df = pd.DataFrame(norm_data, index=history.index, columns=[stock])

            # Update common dates
            if common_dates is None:
                common_dates = set(norm_df.index)
            else:
                common_dates &= set(norm_df.index)

            normalized_data[stock] = norm_df

        # Keep only the common dates
        common_dates = sorted(list(common_dates))
        for stock in self.stocks:
            normalized_data[stock] = normalized_data[stock].loc[common_dates]

        # Concatenate all normalized data
        self.df_norm = pd.concat(normalized_data.values(), axis=1)

        # Split into train and test based on the year
        self.train_data = self.df_norm[self.df_norm.index.year < test_year]
        self.test_data = self.df_norm[self.df_norm.index.year >= test_year]

        # Set data for training or testing
        if self.train:
            self.data = self.train_data
        else:
            # Include last `window` rows of train data in test data
            self.data = pd.concat([self.train_data.tail(self.window), self.test_data])

    def __len__(self):
        """
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data) - self.window

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample.

        Returns:
            tuple: (context_window, target_value) as torch tensors.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of bounds.")

        x = self.data.iloc[index:index + self.window].values
        y = self.data.iloc[index + self.window].values

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)