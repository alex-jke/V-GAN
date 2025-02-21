from torch import Tensor

class PreparedData:
    """
    This class is a simple container for the prepared data.
    """
    def __init__(self, x_train: Tensor, y_train: Tensor, x_test: Tensor, y_test: Tensor, space: str):
        """
        Initializes the PreparedData object.
        :param x_train: The training data as a torch Tensor of shape (batch_size, dim_features).
        :param y_train: The training labels as a torch Tensor of shape (batch_size).
        :param x_test: The test data as a torch Tensor of shape (batch_size, dim_features).
        :param y_test: The test labels as a torch Tensor of shape (batch_size).
        :param space: The space in which the data is prepared.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.space = space
        self._validate_data()

    def _validate_data(self):
        """
        Validates the data.
        """
        if self.x_train.shape[0] != self.y_train.shape[0]:
            raise ValueError(f"The number of training samples and labels must be the same. Got {self.x_train.shape[0]} samples and {self.y_train.shape[0]} labels.")
        if self.x_test.shape[0] != self.y_test.shape[0]:
            raise ValueError(f"The number of test samples and labels must be the same. Got {self.x_test.shape[0]} samples and {self.y_test.shape[0]} labels.")