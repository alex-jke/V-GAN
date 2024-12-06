from text.visualizer.visualizer import Visualizer
import matplotlib.pyplot as plt


class ValueVisualizer(Visualizer):

    def visualize(self, samples: int = 1):
        """
        Visualizes the data with the values.
        :param samples: The number of samples to visualize.
        """
        plot, ax = plt.subplots()

        if samples > 0:
            # Plot sample amount of subspaces
            subspaces = self.get_subspaces(samples)

            for i in range(samples):
                ax.plot(subspaces[i].cpu().numpy(), label=f"Sample {i}")

        # Plot the average subspace
        ax.plot(self.avg_subspace.cpu().numpy(), label="Average")

        ax.set_title("Average Subspace Value per Token position")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Average Subspace Value")

        ax.legend()

        output_path = self.output_dir / "value_visualization.png"
        plt.savefig(output_path)
