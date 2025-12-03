from matplotlib import pyplot as plt
import numpy as np


class BaseVisualizer:
    def __init__(self, figsize: tuple[int]=(10, 5)) -> None:
        self.figsize = figsize

    def plot_timeseries(self, data: np.ndarray, title: str = "Time Series", xlabel: str = "Time", ylabel: str = "Value", legends: list[str] = None) -> None:
        """Plot a simple time series. Data is expected to be a numpty array of shape (N, T)"""
        plt.figure(figsize=self.figsize)
        plt.plot(data.T)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if legends is not None:
            plt.gca().legend(legends)
        plt.grid(True)
        plt.show()