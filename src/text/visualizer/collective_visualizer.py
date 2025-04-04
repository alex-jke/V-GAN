from typing import List

from torch import Tensor

from modules.od_module import VMMD_od, ODModule
from modules.text.vmmd_text_base import VMMDTextBase
from text.Embedding.tokenizer import Tokenizer
from text.visualizer.average_alpha_visualizer import AverageAlphaVisualizer
from text.visualizer.generator_visualizer import GeneratorVisualizer
from text.visualizer.random_alpha_visualizer import RandomAlphaVisualizer

from text.visualizer.subspace.csv_subspace_visualizer import CSVSubspaceVisualizer
from text.visualizer.subspace.distribution_subspace_visualizer import DistributionSubspaceVisualizer
from text.visualizer.value_visualizer import ValueVisualizer
from text.visualizer.visualizer import Visualizer


class CollectiveVisualizer(Visualizer):
    """
    Wraps various visualization tools.
    """
    def __init__(self, tokenized_data: Tensor | List[List[str]], tokenizer: Tokenizer | None, vmmd_model: ODModule, export_path: str, text_visualization: bool = True):
        self.params = {
            "model": vmmd_model,
            "tokenized_data": tokenized_data,
            "tokenizer": tokenizer,
            "path": export_path
        }
        self.text_visualization = text_visualization
        super().__init__(**self.params)

    def visualize(self,samples: int = 1, epoch: int = 0):
        if self.text_visualization:
            avg_vis = AverageAlphaVisualizer(**self.params)
            avg_vis.visualize(samples=samples, epoch=epoch)

            rand_vis = RandomAlphaVisualizer(**self.params)
            rand_vis.visualize(samples=samples, epoch=epoch)

        value_vis = ValueVisualizer(**self.params)
        value_vis.visualize(samples=0, epoch=epoch)
        value_vis.visualize(samples=samples, epoch=epoch)

        csv_subspace_vis = CSVSubspaceVisualizer(**self.params)
        csv_subspace_vis.visualize(samples=samples, epoch=epoch)

        dist_subspace_vis = DistributionSubspaceVisualizer(**self.params)
        dist_subspace_vis.visualize(samples=samples, epoch=epoch)

        generator_visualizer = GeneratorVisualizer(**self.params)
        generator_visualizer.visualize(samples=samples, epoch=epoch)

