from typing import Optional

from numpy import ndarray
from torch import Tensor

from modules.text.vmmd_text_lightning import VMMDTextLightningBase
from text.dataset.dataset import Dataset, AggregatableDataset
from text.dataset_converter.dataset_preparer import DatasetPreparer
from text.outlier_detection.space.prepared_data import PreparedData
from text.outlier_detection.space.space import Space


class WordSpace(Space):
    def transform_dataset(self, dataset: AggregatableDataset, use_cached: bool, inlier_label, masks: Optional[Tensor] = None) -> PreparedData:
        if not isinstance(dataset, AggregatableDataset):
            raise ValueError("WordSpace only works with AggregatableDataset.")
        if use_cached:
            raise ValueError("WordSpace does not support caching.")
        preparer = DatasetPreparer(dataset, self.train_size)
        x_train, y_train = preparer.get_data_with_labels([inlier_label], train=True)
        preparer.max_samples = self.test_size
        x_test, y_test = preparer.get_data_with_labels(train=False)

        # Since we set aggregate to True, the word dimension is only of size 1,
        # but still present so that regardless of the aggregation method, the other models
        # can expect the same input shape.
        embedded_train_with_word_dim = self.model.embed_sentences(x_train, aggregate=True, dataset=dataset)
        embedded_test_with_word_dim = self.model.embed_sentences(x_test, aggregate=True, dataset=dataset)

        assert embedded_train_with_word_dim.shape[1] == 1
        embedded_train = embedded_train_with_word_dim.mean(dim=1)
        embedded_test = embedded_test_with_word_dim.mean(dim=1)

        y_train_int = y_train.astype(int)
        y_test_int = y_test.astype(int)

        y_train_tensor = Tensor(y_train_int.tolist()).int().to(self.model.device)
        y_test_tensor = Tensor(y_test_int.tolist()).int().to(self.model.device)

        return PreparedData(x_train=embedded_train, y_train=y_train_tensor, x_test=embedded_test, y_test=y_test_tensor,
                            space=self.name)

    @property
    def name(self):
        return "Word"

    def get_n_dims(self, x_train: ndarray) -> int:
        """
        Returns the number of dimensions of the space used as features.
        """
        return VMMDTextLightningBase.get_average_sentence_length(x_train)