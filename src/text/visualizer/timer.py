import time
from pathlib import Path
from typing import Dict

from matplotlib import pyplot as plt
from numpy.array_api import linspace


class Timer:
    def __init__(self, amount_epochs: int, export_path: str):
        self.amount_epochs = amount_epochs
        self.start_time = time.time()
        self.times: Dict[int, float] = {}
        self.export_path = Path(export_path) / "times"

    def measure(self, epoch: int):
        cur_time = time.time()
        self.times[epoch] = cur_time - self.start_time

        if epoch >= self.amount_epochs:
            self.export_data()

    def export_data(self):
        x = self.times.keys()
        y = self.times.values()
        plot = plt.plot(x, y)
        plt.xlabel("epoch")
        plt.ylabel("time (s)")
        plt.legend(["epoch", "time (s)"])
        plt.savefig(self.export_path / "times.png")

    def pause(self):
        self.pause_start_time = time.time()

    def resume(self):
        self.start_time = self.start_time + time.time() - self.pause_start_time
