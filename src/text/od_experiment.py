import os
from pathlib import Path
from typing import List

import pandas as pd

from pyod.models.lunar import LUNAR as pyod_LUNAR
from pyod.models.ecod import ECOD as pyod_ECOD
from pyod.models.lof import LOF as pyod_LOF

from text.Embedding.bert import Bert
from text.Embedding.deepseek import DeepSeek1B
from text.Embedding.gpt2 import GPT2
from text.Embedding.huggingmodel import HuggingModel
from text.dataset.ag_news import AGNews
from text.dataset.dataset import Dataset
from text.dataset.emotions import EmotionDataset
from text.dataset.imdb import IMBdDataset
from text.dataset.wikipedia_slim import WikipediaPeopleDataset
from text.outlier_detection.VGAN_odm import VGAN_ODM
from text.outlier_detection.odm import OutlierDetectionModel
from text.outlier_detection.pyod_odm import LOF, LUNAR, ECOD, FeatureBagging
from text.outlier_detection.trivial_odm import TrivialODM
from text.visualizer.result_visualizer import ResultVisualizer

def perform_experiment(dataset: Dataset, model: HuggingModel):
    partial_params = {
        "dataset": dataset,
        "model": model,
        "train_size": 1000, #todo: add ability to not crash when chosen too large
        "test_size": 100,
    }
    params = {**partial_params, **{"use_embedding": True}}
    bases = [pyod_LUNAR, pyod_ECOD, pyod_LOF]
    models: List[OutlierDetectionModel] = ([LOF(**params), LUNAR(**params), ECOD(**params)]
                                           + [FeatureBagging(**partial_params, base_detector=base, use_embedding=True)
                                              for base in bases]
                                           + [FeatureBagging(**partial_params, base_detector=base, use_embedding=False)
                                              for base in bases]
                                           + [VGAN_ODM(**partial_params, base_detector=base, use_embedding=False) for
                                              base in bases]
                                           + [VGAN_ODM(**partial_params, base_detector=base, use_embedding=True) for
                                              base in bases]
                                           + [TrivialODM(**partial_params, guess_inlier_rate=rate) for rate in
                                              [0.0, 0.5, 1.0]])

    result_df = pd.DataFrame()
    dataset._import_data()
    output_path = Path(os.getcwd()) / 'results' / 'outlier_detection_full' / dataset.name / model.model_name
    error_df = pd.DataFrame({"model": []})
    for i, od_model in enumerate(models):
        try:
            od_model.start_timer()
            od_model.train()
            od_model.predict()
            od_model.stop_timer()
            result_df = pd.concat([result_df, od_model.evaluate(output_path)])
        except: #todo: find better solution
            error_df = pd.concat([error_df,
                                  pd.DataFrame({"model": od_model.name})
                                  ])
    print(result_df)
    visualizer = ResultVisualizer(result_df, output_path)
    columns = result_df.columns
    column_names = [column for column in columns if column != "method"]
    [visualizer.visualize(x_column="method", y_column=column) for column in column_names]
    result_df.to_csv(output_path / "results.csv", index=False)
    error_df.to_csv(output_path / "errors.csv", index=False)

if __name__ == '__main__':
    datasets = [EmotionDataset(), IMBdDataset(), AGNews()]
    models = [GPT2(), Bert(), DeepSeek1B()]
    for model in models:
        for dataset in datasets:
            perform_experiment(dataset=dataset, model=model)

