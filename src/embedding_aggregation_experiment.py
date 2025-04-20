import os
import unittest
from argparse import ArgumentError
from pathlib import Path
from typing import Type, List, Optional

import pandas as pd
import torch
from pandas import DataFrame

from text.Embedding.LLM import llms
from text.Embedding.LLM.causal_llm import CausalLLM
from text.Embedding.LLM.huggingmodel import HuggingModel
from text.Embedding.LLM.llama import LLama3B
from text.Embedding.unification_strategy import UnificationStrategy
from text.consts.columns import TYPE_COL, TRAIN_SIZE_COL, TEST_SIZE_COL, EMB_MODEL_COL, PROMPT_COL, RUN_COL
from text.dataset.ag_news import AGNews
from text.dataset.dataset import AggregatableDataset
from text.dataset.emotions import EmotionDataset
from text.dataset.nlp_adbench import NLP_ADBench
from text.dataset.prompt import Prompt
from text.outlier_detection.odm import DATASET_COL
from text.outlier_detection.pyod_odm import LUNAR, BasePyODM, LOF, ECOD
from text.outlier_detection.space.embedding_space import EmbeddingSpace
from text.outlier_detection.space.space import Space
from text.outlier_detection.space.word_space import WordSpace
from text.visualizer.NTPE.csv_visualizer import CsvVisualizer


NTPE = "NPTE"
AVG = "avg"

class EmbeddingAggregationExperiment():

    def __init__(self, *args, **kwargs):
        self.train_size = 15_000
        self.test_size = 3_000
        super().__init__(*args, **kwargs)

    def run_comparison(self, dataset: AggregatableDataset, base: Type[BasePyODM] = LUNAR, model: HuggingModel = None, type: str = NTPE):
        if model is None:
            model = LLama3B()
        train_size = self.train_size
        test_size = self.test_size
        space: Optional[Space] = None
        if type == NTPE:
            space = WordSpace(strategy=UnificationStrategy.TRANSFORMER, model=model, test_size=test_size,
                               train_size=train_size)
        elif type == AVG:
            space = EmbeddingSpace(model=model, train_size=train_size, test_size=test_size)
        else:
            raise ValueError(f"type needs to be either '{AVG}' or '{NTPE}'")

        method = base(
            space=space,
            dataset=dataset,
            use_cached=False
        )

        method.train()
        method.predict()
        metrics, shared_metrics = method.evaluate()
        metrics[TYPE_COL] = type

        train_samples = shared_metrics["total_train_samples"]
        test_samples = shared_metrics["total_test_samples"]

        test_size = int(test_samples.iloc[0])
        train_size = int(train_samples.iloc[0])

        comparison_df = metrics
        comparison_df[DATASET_COL] = dataset.name
        comparison_df[TRAIN_SIZE_COL] = train_size
        comparison_df[TEST_SIZE_COL] = test_size
        comparison_df[EMB_MODEL_COL] = model.model_name
        prompt = dataset.prompt.full_prompt if type == NTPE else ""
        comparison_df[PROMPT_COL] = prompt

        print(comparison_df)

        return comparison_df


    def test_emotions(self):
        dataset = EmotionDataset()
        self.run_comparison(dataset)

    def test_ag_news(self):
        dataset = AGNews()
        self.run_comparison(dataset)

    def run_all(self, output_path: Path, models: List[Type[CausalLLM]]):
        datasets = NLP_ADBench.get_all_datasets()
        datasets.sort(key=lambda d: d.average_length)
        datasets = [dataset for dataset in datasets if dataset.name != NLP_ADBench.sms_spam().name]
        sms_spam_new_prompt = NLP_ADBench.sms_spam()
        sms_spam_old_prompt = NLP_ADBench.sms_spam()
        sms_spam_old_prompt.prompt = Prompt(
            sample_prefix="sms :",
            label_prefix="spam type :",
            samples=["Congratulations! You've won a $1,000 cash prize!",
                     "What is our plan for tonight?"],
            labels=["spam", "no spam"]
        )
        datasets = [sms_spam_old_prompt, sms_spam_new_prompt] + datasets
        amount_runs = 3
        for dataset in datasets:
            for model_cls in models:
                #print(f"Running model {model_cls.__name__}")
                model = model_cls()
                for base in [LUNAR, LOF, ECOD]:
                    for type in [AVG,
                                 NTPE]:
                        for run in range(amount_runs):

                            if output_path.exists():
                                results_df = pd.read_csv(output_path)
                                already_run = results_df[DATASET_COL].tolist()
                                bases = results_df["base"].tolist()
                                emb_models = results_df[EMB_MODEL_COL].tolist()
                                runs_list = results_df[RUN_COL].tolist()
                                prompts = results_df[PROMPT_COL].fillna("no_prompt").tolist()
                                types = results_df[TYPE_COL].tolist()
                                prompt = dataset.prompt.full_prompt if type == NTPE else "no_prompt"
                                if ((dataset.name, base.__name__, model.model_name, run, prompt, type)
                                        in zip(already_run, bases, emb_models, runs_list, prompts, types)):
                                    print(f"Skipping dataset {dataset.name} with base {base.__name__} + run {run} + "
                                          f"type {type}, already present in results.")
                                    continue
                            try:
                                print(f"Running dataset: {dataset.name}, base: {base.__name__}, model: {model.model_name}, run: {run}, type: {type}")
                                result = self.run_comparison(dataset, base, model, type=type)
                                result[RUN_COL] = run
                                result.to_csv(output_path, mode="a", header=not output_path.exists())
                            except Exception as e:
                                print(f"An error occurred running {dataset.name} + {base.__name__}. Skipping. (error: {e}")
                                #raise e
                                continue
                #model.model.to("cpu")
                #del model.model
                del model

    def run_different_prompts(self, dataset: AggregatableDataset, prompts: List[Prompt], output_path: Path, model: HuggingModel):
        default_prompt = dataset.prompt
        prompts = [default_prompt] + prompts
        for prompt in prompts:
            dataset.prompt = prompt
            result = self.run_comparison(dataset, model=model)
            result.to_csv(output_path, mode="a", header=not output_path.exists())

    def compare_sms_spam_prompts(self, output_path: Path, model: HuggingModel):
        dataset = NLP_ADBench.sms_spam()
        prompts = [
            Prompt(
                sample_prefix="sms :",
                label_prefix="spam type :",
                samples=["Congratulations! You've won a $1,000 cash prize!",
                         "What is our plan for tonight?.",
                         "Wht r ur plans 4 2night"],
                labels=["spam", "no spam", "no spam"] #TODO: this yields a little higher auc
            )
        ]
        self.run_different_prompts(dataset, prompts, output_path, model)

    def create_results_csv(self, output_path: Path, experiments: DataFrame):
        visualizer = CsvVisualizer(output_path, experiments)
        visualizer.export_csv()


if __name__ == "__main__":

    exp = EmbeddingAggregationExperiment()

    path = Path(os.path.dirname(__file__)) / "text" / "results" / "aggregation_test"
    path.mkdir(parents=True, exist_ok=True)
    models = llms.get_causal_llms()[8:]

    csv_path = path / "results_small3.csv"

    exp.train_size = 1_000
    exp.test_size = 500

    #exp.run_all(csv_path, models)
    df = pd.read_csv(csv_path)
    output_path = path / "aggregated.csv"
    exp.create_results_csv(output_path, df)

    #self.compare_sms_spam_prompts(output_path, model)






