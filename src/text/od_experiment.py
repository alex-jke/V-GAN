import os
import traceback
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Callable

import pandas as pd
import torch
import random

from pyod.models.lunar import LUNAR as pyod_LUNAR
from pyod.models.ecod import ECOD as pyod_ECOD
from pyod.models.lof import LOF as pyod_LOF

from models.Generator import GeneratorSigmoidAnnealing, GeneratorSoftmaxAnnealing, GeneratorUpperSoftmax, GeneratorSoftmaxSTE, GeneratorSigmoidSTE, GeneratorSpectralSigmoidSTE, GeneratorSigmoidSoftmaxSTE, GeneratorSigmoidSoftmaxSigmoid, GeneratorSigmoidSTEMBD, GeneratorSoftmaxSTESpectralNorm, GeneratorSoftmaxSTEMBD
from text.Embedding.LLM.bert import Bert
from text.Embedding.LLM.deepseek import DeepSeek1B, DeepSeek14B, DeepSeek7B
from text.Embedding.LLM.gpt2 import GPT2
from text.Embedding.LLM.huggingmodel import HuggingModel
from text.Embedding.LLM.llama import LLama1B, LLama3B
from text.Embedding.unification_strategy import UnificationStrategy, StrategyInstance
from text.UI import cli
from text.UI.cli import ConsoleUserInterface
from text.dataset.ag_news import AGNews
from text.dataset.dataset import Dataset
from text.dataset.emotions import EmotionDataset
from text.dataset.imdb import IMBdDataset
from text.dataset.nlp_adbench import NLP_ADBench
from text.dataset.wikipedia_slim import WikipediaPeopleDataset
#from text.outlier_detection import VGAN_odm
#from text.outlier_detection.VGAN_odm import V_ODM, DistanceV_ODM, EnsembleV_ODM
from text.outlier_detection.odm import OutlierDetectionModel
from text.outlier_detection.pyod_odm import LOF, LUNAR, ECOD, FeatureBagging
from text.outlier_detection.space.embedding_space import EmbeddingSpace
from text.outlier_detection.space.token_space import TokenSpace
from text.outlier_detection.space.word_space import WordSpace
from text.outlier_detection.trivial_odm import TrivialODM
from text.outlier_detection.v_method.V_odm import V_ODM
from text.outlier_detection.v_method.distance_v_odm import DistanceV_ODM
from text.outlier_detection.v_method.ensembe_v_odm import EnsembleV_ODM
from text.outlier_detection.v_method.vmmd_adapter import VMMDAdapter
from text.outlier_detection.word_based_v_method.text_v_odm import TextVOdm
from text.result_aggregator import ResultAggregator
from text.visualizer.result_visualizer.rank import RankVisualizer
from text.visualizer.result_visualizer.result_visualizer import ResultVisualizer


class Experiment:
    def __init__(self, dataset: Dataset, emb_model, skip_error: bool = True, train_size: int = -1, test_size: int = -1,
                 models: List[OutlierDetectionModel] = None, output_path: Path = None, experiment_name: str = None,
                 use_cached: bool = False, run_cachable: bool = False, runs=1):
        """
        Initializes the experiment.
        :param dataset: The dataset to use for the experiment.
        :param emb_model: The embedding model to use for the experiment.
        :param skip_error: Whether to skip errors and continue running the experiment.
        :param train_size: The number of training samples to use for the experiment -1 means all samples will be used.
        :param test_size: The number of testing samples to use for the experiment -1 means all samples will be used.
        :param models: A list of OutlierDetectionModel instances to use for the experiment. If None, a default list will be used.
        :param output_path: The path to save the results of the experiment. If None, a default path will be used.
        :param experiment_name: The name of the experiment. If None, a default name will be used.
        :param use_cached: Whether to use cached data for the experiment. Using caching might distort the time measurements.
        :param run_cachable: Whether to run the cachable part of the experiment. If False, the experiment will run all models. This allows for quicker testing.
        :param runs: The number of runs per experiment. This is used for ranking the methods in a box plot.
        """
        """
        Initializes the experiment.
        """
        self.inlier_label = dataset.get_possible_labels()[0]
        self.dataset = dataset
        self.emb_model = emb_model
        self.skip_error = skip_error
        self.experiment_name = "experiment_od" if experiment_name is None else experiment_name
        self.use_cached = use_cached
        self.run_cachable = run_cachable
        self.runs = runs

        # Experiment parameters
        self.token_params: Dict = {
            "dataset": self.dataset,
            "use_cached": use_cached,
            "inlier_label": self.inlier_label,
            "space": TokenSpace(model=emb_model, train_size=train_size, test_size=test_size)
        }
        self.emb_params: Dict = {
            "dataset": self.dataset,
            "use_cached": use_cached,
            "inlier_label": self.inlier_label,
            "space": EmbeddingSpace(model=emb_model, train_size=train_size, test_size=test_size)
        }

        text_base = {
            "dataset": self.dataset,
            "use_cached": use_cached,
            "inlier_label": self.inlier_label,}
        self.text_params: Dict[UnificationStrategy,Dict] = {
            UnificationStrategy.TRANSFORMER: {
                **text_base,
                "space": WordSpace(model=emb_model, train_size=train_size, test_size=test_size, strategy = UnificationStrategy.TRANSFORMER)
            },
            UnificationStrategy.MEAN: {
                **text_base,
                "space": WordSpace(model=emb_model, train_size=train_size, test_size=test_size,
                                   strategy=UnificationStrategy.MEAN)
            }
        }

        # Determine the output directory.
        self.output_path = output_path
        if output_path is None:
            self.output_path: Path = self._get_output_path()

        # Build the list of outlier detection models.
        self.models = models
        if models is None:
            self.models: List = self._build_models()

        # DataFrames to store experiment results and errors.
        self.result_df: pd.DataFrame = pd.DataFrame()
        self.comon_metrics: pd.DataFrame = pd.DataFrame()
        self.error_df: pd.DataFrame = pd.DataFrame(columns=["model", "error"])

        self.ui = cli.get()

    @classmethod
    @property
    def result_csv_name(cls) -> str:
        return "results.csv"

    @classmethod
    @property
    def comon_metrics_name(cls) -> str:
        return "comon_metrics.csv"

    def _build_models(self) -> List[OutlierDetectionModel]:
        """
        Builds and returns a list of outlier detection model instances.
        """
        bases = [
            pyod_LUNAR,
            #pyod_ECOD,
            pyod_LOF]
        models = []

        # Base models that always use embedding or text.
        models.extend([
            LOF(**self.emb_params),
            LUNAR(**self.emb_params),
            ECOD(**self.emb_params)
        ])

        if not self.run_cachable and False:
            # VMMD Text ODM with subspace distance and ensemble outlier detection.
            models.extend([TextVOdm(**self.text_params[strategy], base_detector=base, output_path=self.output_path, aggregation_strategy=strategy.create())
                           for base in bases
                           for strategy in [UnificationStrategy.TRANSFORMER, UnificationStrategy.MEAN]])
            # VMMD Text ODM with only ensemble outlier detection.
            models.extend([TextVOdm(**self.text_params[strategy], base_detector=base, output_path=self.output_path, subspace_distance_lambda=0.0, aggregation_strategy=strategy.create())
                           for base in bases
                           for strategy in [UnificationStrategy.TRANSFORMER, UnificationStrategy.MEAN]])
            # VMMD Text ODM with only subspace distance.
            models.extend([TextVOdm(**self.text_params[strategy], base_detector=base, output_path=self.output_path, classifier_delta=0.0, aggregation_strategy=strategy.create())
                            for base in bases
                            for strategy in [UnificationStrategy.TRANSFORMER, UnificationStrategy.MEAN]])


        # Trivial ODM models with different guess inlier rates.
        models.extend([
            TrivialODM(**self.token_params, base_method=base)
            for base in bases
        ])


        # VGAN ODM models with both use_embedding False and True.
        params = [self.emb_params, self.text_params[UnificationStrategy.TRANSFORMER], self.text_params[UnificationStrategy.MEAN]] + ([] if self.run_cachable else [self.token_params])
        model_types = [lambda generator: VMMDAdapter(generator=generator, export_generator=self.use_cached)]#, VGANAdapter()]
        generators = [#GeneratorSigmoidAnnealing, #GeneratorSoftmaxAnnealing,
                      GeneratorUpperSoftmax,
                      GeneratorSoftmaxSTE, GeneratorSigmoidSTE#, GeneratorSpectralSigmoidSTE, GeneratorSigmoidSoftmaxSTE, GeneratorSigmoidSoftmaxSigmoid,
                      #GeneratorSigmoidSTEMBD,
                      #GeneratorSoftmaxSTESpectralNorm,
                      #GeneratorSoftmaxSTEMBD
                      ]

        # VGAN ODM with both ensemble outlier detection, and subspace distance, only using pre-embedded, as euclidian
        # distance does not make sense for tokens.
        models.extend(
            [V_ODM(**param, base_detector=base,
                   output_path = self.output_path, odm_model=model_type(generator=generator))
             for base in bases
             for model_type in model_types
             for param in params
             for generator in generators]
        )

        # VGAN ODM on the token space only using the ensemble method.
        if not self.run_cachable:
            models.extend([EnsembleV_ODM(**self.token_params, base_detector=base,
                                         output_path=self.output_path) for base in bases])


        # VGAN ODM with only ensemble outlier detection.
        models.extend([EnsembleV_ODM(**param, output_path=self.output_path, odm_model=model_type(generator))
                       for param in params
                       for model_type in model_types
                       for generator in generators
                       ])

        # VGAN ODM with only subspace distance.
        models.extend([DistanceV_ODM(**param, output_path=self.output_path,
                                     odm_model=model_type(generator))
                       for model_type in model_types
                       for param in params
                       for generator in generators
                       ])

        models.extend([
            FeatureBagging(**param, base_detector=base)
            for base in bases
            for param in params
        ])

        for strategy in [UnificationStrategy.TRANSFORMER, UnificationStrategy.MEAN]:
            models.extend([
                LOF(**self.text_params[strategy]),
                LUNAR(**self.text_params[strategy]),
                ECOD(**self.text_params[strategy]),
            ])

        return models


    def _get_output_path(self) -> Path:
        """
        Constructs and returns the output path for saving results.
        """
        return Path(os.getcwd()) / 'results' / "outlier_detection" /self.experiment_name / self.dataset.name / self.emb_model.model_name

    def _run_single_model(self, model) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Runs training, prediction, and evaluation for a single model.
        Returns a tuple (evaluation_results, error_record) where each is a DataFrame.
        """
        try:
            #model.start_timer()
            #if self.emb_model.model_name == DeepSeek1B().model_name and "VMMD + LUNAR + T" in model._get_name():
                #raise Exception("Skipping DeepSeek1B with VMMD + LUNAR + T, due to high runtime")
            model.train()
            model.predict()
            #model.stop_timer()
            evaluation, self.comon_metrics = model.evaluate(self.output_path)
            print(f" | finished successfully (auc: {float(evaluation['auc']):>1.3f}).")
            return evaluation, None
        except Exception as e:
            if not self.skip_error:
                raise e
            error_record = pd.DataFrame({
                "model": [model._get_name()],
                "error": [str(e)],
                "traceback": [str(traceback.format_exc())]
            })
            print(f"{model._get_name()} encountered an error.")
            return pd.DataFrame(), error_record

    def _visualize_and_save_results(self, run: int) -> None:
        """
        Visualizes the results and saves both the result and error DataFrames to CSV files.
        """
        run_dir = self._get_run_path(run)
        visualizer = ResultVisualizer(self.result_df, output_dir=run_dir)
        for column in self.result_df.columns:
            if column != "method":
                visualizer.visualize(x_column="method", y_column=column)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.result_df.to_csv(run_dir / self.result_csv_name, index=False)
        self.error_df.to_csv(run_dir / "errors.csv", index=False)
        if not (run_dir / self.comon_metrics_name).exists():
            self.comon_metrics.to_csv(run_dir / self.comon_metrics_name, index=False)

    def _get_run_path(self, run: int) -> Path:
        return self.output_path / f"run_{run}"

    def _filter_for_not_run(self, run: int) -> None:
        self.models = self._build_models()
        result_path = self._get_run_path(run) / self.result_csv_name
        if not result_path.exists():
            return
        try:
            result_df = pd.read_csv(result_path)
        except pd.errors.EmptyDataError:
            return
        not_run = []
        for model in self.models:
            run = result_df[model.method_column].tolist()
            if model._get_name() not in run:
                not_run.append(model)
        #print(f"Not run: {len(not_run)} models")
        random.shuffle(not_run)
        self.models = not_run
        self.result_df = result_df


    def run(self) -> pd.DataFrame:
        """
        Executes the complete experiment pipeline.
        """
        # Import dataset data.
        self.dataset._import_data()  # Consider renaming to a public method if possible.

        results: List[pd.DataFrame] = []

        with self.ui.display():
            for run in range(self.runs):
                self.result_df = pd.DataFrame(columns=self.result_df.columns)
                self.ui.update(f"Run {run + 1}/{self.runs}")
                self._filter_for_not_run(run=run)

                # Run each model experiment.
                with self.ui.display():
                    for model in self.models:
                        self.ui.update(f"Running model {model._get_name()}")
                        evaluation, error = self._run_single_model(model)
                        self.result_df = pd.concat([self.result_df, evaluation], ignore_index=True)
                        if error is not None:
                            self.error_df = pd.concat([self.error_df, error], ignore_index=True)
                            if not self.output_path.exists():
                                self.output_path.mkdir(parents=True, exist_ok=True)
                            self.error_df.to_csv(self.output_path / "errors.csv", index=False)
                            continue
                        self._visualize_and_save_results(run=run)
                        del model
                    if len(self.result_df) > 0:
                        results.append(self.result_df)
                        self._visualize_ranks(results)

                self._visualize_and_save_results(run=run)
        aggregated_results = pd.concat(results, ignore_index=True)
        return aggregated_results

    def _visualize_ranks(self, results: List[pd.DataFrame]):
        visualizer = RankVisualizer(results, output_dir=self.output_path)
        metrics = ["auc", "prauc", "f1"]
        for metric in metrics:
            visualizer.visualize(method_col="method", metric_col=metric, group_by="base")
        print("Ranking visualizations saved to", self.output_path)

def aggregate_results():
    aggregator = ResultAggregator(
        common_metrics_name=Experiment.comon_metrics_name,
        result_name=Experiment.result_csv_name
    )
    aggregator.run_aggregation()

MODEL = "model"
DATASET = "dataset"

if __name__ == '__main__':
    """datasets = [
                   AGNews(),
                   EmotionDataset(),
                   IMBdDataset(),
                   ] + NLP_ADBench.get_all_datasets()
    embedding_models = [#DeepSeek1B,
                        GPT2, Bert,
                        #DeepSeek7B
                        ]

    # Python garbage collection does not collect garbage, that is clear the created models and datasets.
    # Thus, eventually, the program will crash.
    # The Experiment class checks, which models and datasets have been run. This approach guarantees, that
    # every model and dataset can run at some point.
    random.shuffle(embedding_models)
    #random.shuffle(datasets)

    ui = cli.get()
    train_size = 5_000
    test_size = 1_000

    # Create and run an experiment for every combination of dataset and embedding model.
    with ui.display():
        for dataset in datasets:
            ui.update(f"dataset {dataset.name}")
            with ui.display():
                for emb_model_cls in embedding_models:
                    emb_model = emb_model_cls()
                    ui.update(f"embedding model {emb_model.model_name}")
                    experiment = Experiment(dataset=dataset, emb_model=emb_model, train_size=train_size, test_size=test_size,
                                            experiment_name=f"0.29_smaller", use_cached=True,
                                            run_cachable=False, skip_error=True, runs=5)
                    experiment.run()
                    aggregate_results()
                    del emb_model
                    del experiment

                    if torch.cuda.is_available():
                        memory_used_before = torch.cuda.memory_allocated()
                        torch.cuda.empty_cache()
                        memory_used_after = torch.cuda.memory_allocated()
                        print(f"freed cuda cache: {memory_used_before} -> {memory_used_after}")"""
    test_samples = 3000
    train_samples = 15_000
    datasets = NLP_ADBench.get_all_datasets()
    datasets.sort(key=lambda d: d.average_length)
    emb_model = LLama3B()
    for dataset in datasets:
        exp = Experiment(dataset, emb_model, skip_error=True, train_size=train_samples, test_size=test_samples,
                            experiment_name="0.41", use_cached=True, runs=5, run_cachable=True)
        aggregated_path = exp.output_path.parent.parent # directory of the current version
        csv_path = aggregated_path / "aggregated.csv"
        results = exp.run()
        results[MODEL] = emb_model.model_name
        results[DATASET] = dataset.name
        results.to_csv(csv_path , index=False, header= not csv_path.exists())
