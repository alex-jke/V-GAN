import os
from datetime import datetime
from typing import List, Type, Optional

from numpy import ndarray
from transformers import GPT2Model

from VMMDBase import VMMDBase
from models.Generator import GeneratorSigmoidSTE, Generator_big, GeneratorSoftmaxSTE, GeneratorUpperSoftmax, \
    GeneratorSoftmax, GeneratorSoftmaxSTEMBD, Generator, GeneratorSigmoidSoftmaxSTE, GeneratorSigmoidSoftmaxSigmoid, \
    GeneratorSoftmaxSTESpectralNorm, GeneratorSpectralSigmoidSTE
from modules.text.vmmd_text import VmmdText
from modules.text.vmmd_text_preembed import VMMDTextPreEmbed
from text.Embedding.deepseek import DeepSeek1B
from text.Embedding.embedding import Embedding
from text.Embedding.fast_text import FastText
from text.Embedding.gpt2 import GPT2
from text.Embedding.llama import LLama1B
from text.dataset.ag_news import AGNews
from text.dataset.dataset import Dataset
from text.dataset.emotions import EmotionDataset
from text.dataset_converter.dataset_preparer import DatasetPreparer
from text.v_experiment import VBaseExperiment
from text.visualizer.collective_visualizer import CollectiveVisualizer
from modules.text.vmmd_text_base import VMMDTextBase


class VMMDTextExperiment:

    def __init__(self, dataset: Dataset, version: str, samples: int = -1, sequence_length: int | None = None, train: bool = False, epochs: int = 2000,
                 penalty_weight: float = 0.1, batch_size: int = 2000, weight_decay = 0, generator: Generator_big = GeneratorSigmoidSTE,
                 lr: float = 10e-5, gradient_clipping: bool = False, emb_model: Embedding = FastText(normalize=True), transformer_aggregation: bool = False,
                 v_method: Type[VMMDTextBase] = VMMDTextPreEmbed, yield_epochs: Optional[int] = None ):
        self.dataset = dataset
        self.version = version
        self.samples = samples
        self.sequence_length = sequence_length
        self.train = train
        self.epochs = epochs
        self.penalty_weight = penalty_weight
        self.generator = generator
        self.emb_model: Embedding = emb_model
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.lr = lr
        self.gradient_clipping = gradient_clipping
        self.v_method: Type[VMMDTextBase] = v_method
        self.transformer_aggregation = transformer_aggregation
        self.yield_epochs = yield_epochs
        if self.yield_epochs is None:
            self.yield_epochs = self.epochs // 20
        self.export_path = self._build_export_path()
        if dataset.name != EmotionDataset().name:
            raise NotImplementedError("Remove hardcoded postfix.")

    def _get_name(self) -> str:
        return "VMMD_Text"

    def run(self):
        model = self.v_method(print_updates=True, path_to_directory=self.export_path, epochs=self.epochs, weight=self.penalty_weight,
                             sequence_length=self.sequence_length, batch_size=self.batch_size, weight_decay=self.weight_decay,
                             generator=self.generator, lr=self.lr, gradient_clipping=self.gradient_clipping)
        embedding_fun = lambda samples, padding_length, masks: self.emb_model.embed_sentences(samples, padding_length, masks=masks, aggregate=self.transformer_aggregation)
        preparer = DatasetPreparer(self.dataset, max_samples=self.samples)
        x_train = preparer.get_training_data()
        for epoch in model.yield_fit(x_train, embedding_fun, yield_epochs=self.yield_epochs):
            self.visualize(epoch, model, x_train)
        self.visualize(self.epochs, model, x_train)

    def visualize(self, epoch: int, model, sentences: ndarray):
        samples = 30
        sentences: List[List[str]] = [self.emb_model.get_words(sentence) for sentence in sentences[:samples]]
        visualizer = CollectiveVisualizer(tokenized_data=sentences, tokenizer=None, vmmd_model=model,
                                          export_path=self.export_path, text_visualization=True)
        visualizer.visualize(epoch=epoch, samples=30)
        model._export(model.generator, export_params=False)

    def _build_export_path(self) -> str:
        sl_str = self.sequence_length if self.sequence_length is not None else "(avg)"
        base_dir = os.path.join(
            os.getcwd(),
            'experiments',
            "VMMD_Text",
            self.emb_model.__class__.__name__,
            self.generator.__name__,
            f"{self.version}",
            f"agg_" + "t" if self.transformer_aggregation else "avg",
            f"{self.dataset.name}_sl{sl_str}_s{self.samples}"
        )
        if self.train:
            base_dir += "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        return base_dir

def softmax_experiment():
    params = {"version": "0.15+exp_penalty", "train": True, "epochs": 50, "penalty_weight": 0,
     "samples": 500,
     "weight_decay": 0 , "generator": GeneratorSoftmaxSTE, "lr": 0.1,
     "gradient_clipping": False,
     "emb_model": LLama1B(), "v_method": VmmdText, "transformer_aggregation": True, "yield_epochs": 5,
     "batch_size": 100
              }
    VMMDTextExperiment(dataset=EmotionDataset(), **params).run()

def softmax_spectral_norm_experiment():
    params = {"version": "0.15+no_bn+leaky_relu", "train": True, "epochs": 200, "penalty_weight": 0,
     "samples": 500,
     "weight_decay": 0.0 , "generator": GeneratorSoftmaxSTESpectralNorm, "lr": 0.01,
     "gradient_clipping": False,
     "emb_model": LLama1B(), "v_method": VmmdText, "transformer_aggregation": True, "yield_epochs": 5,
     "batch_size": 100
              }
    VMMDTextExperiment(dataset=EmotionDataset(), **params).run()

def sigmoid_experiment():
    #for ta in [True, False]:
        #for pw in [0.1]:
            #for gc in [True, False]:
                #for epochs, lr in [(50, 0.1), (100, 1e-2), (200, 1e-3)]:
    params = {"version": "0.15+exp_penalty+DyT+ReLU", "train": True, "epochs": 200, "penalty_weight": 500,
                      "samples": 500,
                      "weight_decay": 0.0, "generator": GeneratorSigmoidSTE, "lr": 0.05,
                      "gradient_clipping": False,
                      "emb_model": LLama1B(), "v_method": VmmdText, "transformer_aggregation": True, "yield_epochs": 1,
                      "batch_size": 100
                      }
    VMMDTextExperiment(dataset=EmotionDataset(), **params).run()

def spectral_sigmoid_experiment():
    params = {"version": "0.151+only_div_penalty+DyT+ReLU", "train": True, "epochs": 2000, "penalty_weight": 10000.0,
              "samples": 500,
              "weight_decay": 0.0, "generator": GeneratorSpectralSigmoidSTE, "lr": 1e-2,
              "gradient_clipping": False,
              "emb_model": LLama1B(), "v_method": VmmdText, "transformer_aggregation": True, "yield_epochs": 200,
              "batch_size": 100
              }
    VMMDTextExperiment(dataset=EmotionDataset(), **params).run()


if __name__ == '__main__':
    """
    params_sig = {"version":"0.139_sigmoid+16_latent", "train":False, "epochs":5_000, "penalty_weight":0.0, "samples":10_000, "weight_decay":0, "generator": GeneratorSigmoidSTE, "lr":5e-4, "gradient_clipping":False}
    params_soft = {"version":"0.139+larger_betas(adam)+sigmoid_act+no_batchnorm", "train":False, "epochs":4_000, "penalty_weight":0, "samples":10_000,
            "weight_decay":0, "generator": GeneratorSigmoidSoftmaxSTE, "batch_size": 1000, "lr":1e-4, "gradient_clipping":False}
    #for params in [params_sig, params_soft]:
    datasets = ([
                   EmotionDataset(), AGNews(),
                   IMBdDataset()
                ] +
                NLP_ADBench.get_all_datasets()[:1] +
                NLP_ADBench.get_all_datasets()[2:]
                + [WikipediaPeopleDataset(), ])
    for dataset in datasets:
        experiment = VMMDTextExperiment(dataset=dataset, **params_sig)
        experiment.run()"""
    """for penalty_weight in [0.0, 0.2]:
        for lr, epochs in [(0.1, 50), (5e-2, 50), (1e-2, 50)]:#, (1e-3, 200), (1e-4, 500), (1e-5, 1000)]:
            for weight_decay in [1e-3, 1e-4, 1e-5, 0]:
                params_sig = {"version":"0.144sigmoid", "train":True, "epochs":epochs, "penalty_weight":penalty_weight, "samples":3000, "weight_decay":weight_decay, "generator": GeneratorSigmoidSTE, "lr":lr, "gradient_clipping":True,
                      "emb_model": GPT2(), "v_method": VmmdText, "batch_size": 750, "yield_epochs": 5
                      }
                VMMDTextExperiment(dataset=EmotionDataset(), **params_sig).run()
    """
    """for gradient_clipping in [False, True]:
        for penalty_weight in [0.2, 0.5]:
            for lr, epochs in [(0.1, 50), (1e-2, 100), (1e-3, 200)]:  # , (1e-3, 200), (1e-4, 500), (1e-5, 1000)]:
                for weight_decay in [1e-3, 1e-4, 1e-5, 0]:
                    params_sig = {"version":"0.146_grid", "train":True, "epochs":epochs, "penalty_weight":penalty_weight, "samples":5000, "weight_decay":weight_decay, "generator": GeneratorSigmoidSTE, "lr":lr, "gradient_clipping":gradient_clipping,
                                  "emb_model": LLama(), "v_method": VmmdText, "transformer_aggregation": True, "yield_epochs": 5, "batch_size": 250
                                  }
                    VMMDTextExperiment(dataset=EmotionDataset(), **params_sig).run()"""
    """params_softmax = {"version": "0.148", "train": True, "epochs": 100, "penalty_weight": 0.0, "samples": 5000,
     "weight_decay": 1e-5, "generator": GeneratorUpperSoftmax, "lr": 1e-3, "gradient_clipping": False,
     "emb_model": LLama(), "v_method": VmmdText, "transformer_aggregation": True, "yield_epochs": 5, "batch_size": 250
     }"""
    """for gradient_clipping in [False, True]:
        for penalty_weight in [100, 2, 0.5]:
            for lr, epochs in [(0.1, 50), (1e-2, 100), (1e-3, 200), (1e-4, 500), (1e-5, 1000)]:
                params_sigmoid = {"version": "0.148+exp_penalty", "train": True, "epochs": epochs, "penalty_weight": penalty_weight, "samples": 500,
                 "weight_decay": 1e-5, "generator": GeneratorSigmoidSTE, "lr": lr, "gradient_clipping": gradient_clipping,
                 "emb_model": LLama(), "v_method": VmmdText, "transformer_aggregation": True, "yield_epochs": 1, "batch_size": 250
                 }
                #params = [params_softmax, params_sigmoid]
                #for params in params:
                VMMDTextExperiment(dataset=EmotionDataset(), **params_sigmoid).run()"""

    """params_sigmoid = {"version": "0.148+exp_penalty", "train": True, "epochs": 200, "penalty_weight": 50_000,
                      "samples": 5000,
                      "weight_decay": 0.0, "generator": GeneratorSigmoidSTE, "lr": 0.1,
                      "gradient_clipping": False,
                      "emb_model": LLama(), "v_method": VmmdText, "transformer_aggregation": True, "yield_epochs": 10,
                      "batch_size": 250
                      }"""
    """params_sigmoid = {"version": "0.148+exp_penalty", "train": True, "epochs": 50, "penalty_weight": 0,
                      "samples": 500,
                      "weight_decay": 1e-5, "generator": GeneratorSoftmax, "lr": 0.1,
                      "gradient_clipping": False,
                      "emb_model": LLama(), "v_method": VmmdText, "transformer_aggregation": True, "yield_epochs": 1,
                      "batch_size": 250
                      }"""
    # params = [params_softmax, params_sigmoid]
    # for params in params:
    #for penalty_weight in [50_000, 200_000, 1_000_000, 40_000]:
        #for lr in [0.1, 1e-2, 1e-3]:
            #params_sigmoid = {"version": "0.149+exp_penalty", "train": True, "epochs": 200, "penalty_weight": penalty_weight,
            #                  "samples": 5000,
            #                  "weight_decay": 0.0, "generator": GeneratorSigmoidSTE, "lr": lr,
            #                  "gradient_clipping": False,
            #                  "emb_model": LLama(), "v_method": VmmdText, "transformer_aggregation": True, "yield_epochs": 10,
            #                  "batch_size": 250
            #                  }
            #VMMDTextExperiment(dataset=EmotionDataset(), **params_sigmoid).run()
    #softmax_experiment()
    #softmax_spectral_norm_experiment()
    #sigmoid_experiment()
    spectral_sigmoid_experiment()
