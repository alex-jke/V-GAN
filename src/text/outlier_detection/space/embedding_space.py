from typing import Optional

import torch
from torch import Tensor

from text.Embedding.LLM.huggingmodel import HuggingModel
from text.dataset.dataset import Dataset
from text.dataset_converter.dataset_embedder import DatasetEmbedder
from text.outlier_detection.space.prepared_data import PreparedData
from text.outlier_detection.space.space import Space
from text.outlier_detection.space.token_space import TokenSpace

name = "Embedding"
class EmbeddingSpace(Space):

    @property
    def name(self):
        return name

    def transform_dataset(self, dataset: Dataset, use_cached: bool, inlier_label, mask: Optional[Tensor]) -> PreparedData:
        if use_cached:
            prepared_data = self._use_cached(dataset, inlier_label)
        else:
            prepared_data = self._create_data(dataset, inlier_label)
        if mask is None:
            return prepared_data

        # Apply the mask to the data
        mask = mask.to(prepared_data.x_train.device)
        assert mask.dtype == torch.int, f"Mask should be of type int, but got {mask.dtype}"
        assert len(mask.shape) == 1, f"Mask should be 1D, but got {mask.shape}"
        bool_mask = mask == 1

        prepared_data.x_train = prepared_data.x_train[: bool_mask]
        prepared_data.x_test = prepared_data.x_test[: bool_mask]

        return prepared_data

    def _use_cached(self, dataset: Dataset, inlier_label) -> PreparedData:
        dataset_embedder = DatasetEmbedder(dataset, self.model)

        x_train, y_train = dataset_embedder.embed(train=True, samples=self.train_size,
                                                              labels=[inlier_label])

        x_test, y_test = dataset_embedder.embed(train=False, samples=self.test_size)

        return PreparedData(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, space=name, inlier_labels=[inlier_label])

    def _create_data(self, dataset: Dataset, inlier_label) -> PreparedData:
        # Get tokenized data and corresponding labels
        token_space = TokenSpace(self.model, self.train_size, self.test_size)
        #token_data = token_space.transform_dataset(dataset, use_cached=False, inlier_label=inlier_label)

        x_train, y_train, x_test, y_test = token_space.get_tokenized(dataset, inlier_label)

        x_train = token_space.embed_tokenized(x_train)
        x_test = token_space.embed_tokenized(x_test)
        return PreparedData(x_train=x_train,
                            y_train=y_train,
                            x_test=x_test,
                            y_test=y_test,
                            space=name,
                            inlier_labels=[inlier_label])

    def __prepare_embedding(self, tokenized: Tensor) -> Tensor:
        embedding_func = self.model.get_embedding_fun(batch_first=True)
        embedded = embedding_func(tokenized)
        means = embedded.mean(1, keepdim=True)
        stds = embedded.std(1, keepdim=True)
        standardized = (embedded - means) / stds
        normalized = torch.nn.functional.normalize(standardized, p=2, dim=1)
        return normalized

