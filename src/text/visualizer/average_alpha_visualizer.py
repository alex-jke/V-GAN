from text.visualizer.alpha_visualizer import AlphaVisualizer


class AverageAlphaVisualizer(AlphaVisualizer):

    def visualize(self, samples: int = 1, epoch: int = -1):
        """
        Visualizes the data with alpha characters.
        :param samples: The number of samples to visualize.
        """

        # create samples amount of the average subspace in a tensor.
        subspaces = [self.avg_subspace] * samples
        sample_data = self.tokenized_data[:samples]

        self.export_html(sample_data, subspaces, "avg", epoch)