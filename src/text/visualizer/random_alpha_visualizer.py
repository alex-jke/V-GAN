from text.visualizer.alpha_visualizer import AlphaVisualizer


class RandomAlphaVisualizer(AlphaVisualizer):

    def visualize(self, samples: int = 1, epoch: int = -1):
        """
        Visualizes the data with random alpha characters.
        :param samples: The number of samples to visualize.
        """

        # create samples amount of random subspaces in a tensor.
        subspaces = self.get_subspaces(samples)
        sample_data = self.tokenized_data[:samples]

        self.export_html(sample_data, subspaces, "rand", epoch)