from colors import VGAN_GREEN
from text.visualizer.visualizer import Visualizer
import matplotlib.pyplot as plt
import numpy as np

class GeneratorVisualizer(Visualizer):
    def visualize(self, samples: int = 1, epoch: int = 0):
        gen_values = self.model.generator.avg_mask.value.detach().cpu().numpy()
        x = np.linspace(1, len(gen_values), len(gen_values))
        bars = plt.bar(x=x, height=gen_values)
        for bar in bars:
            bar.set_color(VGAN_GREEN)
        plt.xlabel('Dimension')
        plt.ylabel("Activation")
        plt.title('Generator Activation before the final activation')
        output_dir = self.output_dir / 'generator'
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f'activation_{epoch}.png')
        plt.close()