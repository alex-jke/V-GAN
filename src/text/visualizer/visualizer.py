from abc import ABC


class Visualizer(ABC):
    """
    Abstract class for visualizing data.
    """

    def __init__(self):
        pass

    def visualize(self, samples: int = 1):
        """
        Visualizes the data.
        :param text: The text to visualize.
        """
        raise NotImplementedError