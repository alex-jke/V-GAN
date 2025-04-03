from text.dataset.dataset import Dataset, AggregatableDataset
from text.dataset_converter.dataset_preparer import DatasetPreparer
from text.outlier_detection.space.prepared_data import PreparedData
from text.outlier_detection.space.space import Space


class WordSpace(Space):
    def transform_dataset(self, dataset: AggregatableDataset, use_cached: bool, inlier_label) -> PreparedData:
        if not isinstance(dataset, AggregatableDataset):
            raise ValueError("WordSpace only works with AggregatableDataset.")
        preparer = DatasetPreparer(dataset, self.train_size)
        x_train, y_train = preparer.get_data_with_labels([inlier_label], train=True)
        preparer.max_samples = self.test_size
        x_test, y_test = preparer.get_data_with_labels(train=False)
        return PreparedData(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                            space=self.name, aggregetable=dataset)

    @property
    def name(self):
        return "Word"