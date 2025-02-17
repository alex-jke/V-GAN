import os

from models.Generator import GeneratorSigmoidSTE, GeneratorUpperSoftmax, GeneratorSigmoid
from modules.od_module import VGAN_od
from text.Embedding.deepseek import DeepSeek1B
from text.Embedding.gpt2 import GPT2
from text.dataset.ag_news import AGNews
from text.dataset.emotions import EmotionDataset
from text.dataset_converter.dataset_embedder import DatasetEmbedder
from text.dataset_converter.dataset_tokenizer import DatasetTokenizer
from text.v_experiment import VExperiment
from text.visualizer.collective_visualizer import CollectiveVisualizer
from vgan import VGAN

class VGANExperiment(VExperiment):
    """
    Encapsulates one experiment for VGAN (data processing, model training/evaluation, visualization).
    """

    def _get_model(self):
        return VGAN_od(path_to_directory=self.export_path, epochs=self.epochs,  batch_size=self.batch_size,
                       lr_G=self.lr, lr_D=self.lr * 10, weight_decay=self.weight_decay,  seed=None,
                       penalty=self.penalty_weight, generator=self.generator_class, print_updates=True,
                       gradient_clipping= self.gradient_clipping
        )

    def _get_name(self) -> str:
        return "VGAN"

if __name__ == "__main__":
    dataset = EmotionDataset()
    model = DeepSeek1B()
    epochs = 10_000
    yield_epochs = epochs // 20
    exp = VGANExperiment(dataset, model, epochs=epochs, yield_epochs=yield_epochs,
                         pre_embed=True, version="0.11_test_collapse", lr=10e-5, train=True, samples=10_000, generator_class=GeneratorSigmoidSTE,
                         weight_decay=10e-5)
    exp.run()
    #lr_G=0.01, lr_D=0.1