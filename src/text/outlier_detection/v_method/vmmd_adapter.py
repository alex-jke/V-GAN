from pathlib import Path

from numpy import ndarray

from models.Generator import GeneratorSigmoidSTE, GeneratorSigmoidAnnealing, GeneratorUpperSoftmax
from modules.od_module import VMMD_od, VGAN_od
from text.outlier_detection.space.prepared_data import PreparedData
from text.outlier_detection.space.space import Space
from text.outlier_detection.space.token_space import TokenSpace
from text.outlier_detection.v_method.numerical_v_adapter import NumericalVOdmAdapter


class VMMDAdapter(NumericalVOdmAdapter):
    """
    Adapter for the VMMD model used for outlier detection.
    """

    def __init__(self,  seed = 777,
                        epochs = 5_000,
                        lr = 3e-4,
                        penalty_weight = 0.01,
                        weight_decay = 1e-3,
                        generator = GeneratorUpperSoftmax,
                        dataset_specific_params = False,
                        max_batch_size = 3000,
                        export_generator: bool = False
                 ):
        """
        The constructor for the VMMDAdapter class. It allows setting up the VMMD model.
        :param seed: random seed for the model to use.
        :param epochs: number of epochs to train the model. This is overwritten, if dataset_specific_params is True.
        :param lr: learning rate for the model.
        :param penalty_weight: penalty weight for the model.
        :param weight_decay: weight decay for the model.
        :param generator: generator for the model.
        :param dataset_specific_params: whether to use dataset specification or not. For example, to recalculate the
            amount of epochs, as smaller datasets require more epochs.
        :param max_batch_size: maximum batch size to use.
        :param export_generator: A bool whether to export the params of the generator. If set to true, the generator
            will also be reused within the same experiment.
        """
        self.seed = seed
        self.epochs = epochs
        self.lr = lr
        self.penalty_weight = penalty_weight
        self.weight_decay = weight_decay
        self.generator = generator
        self.dataset_specific_params = dataset_specific_params
        self.max_batch_size = max_batch_size
        self.export_generator = export_generator
        super().__init__()

    def _init_model(self, data: PreparedData, space: Space) -> VMMD_od:
        if self.dataset_specific_params:
            self._update_params(data, space)
        model = VMMD_od(penalty_weight=self.penalty_weight, generator=self.generator,
                        lr=self.lr, epochs=self.epochs, seed=self.seed, path_to_directory=self.output_path,
                        weight_decay=self.weight_decay, batch_size=self.max_batch_size, export_generator=self.export_generator)


        return model

    def _update_params(self, data: PreparedData, space: Space):
        """
        Updates the parameters of the model based on the size of the dataset.
        This is done, as the model converges faster on larger datasets.
        Adjusts the number of epochs, so the model can train longer on smaller datasets.
        Adjusts the batch size, as mmd test requires linearly growing batch size with number of dimensions to stay statistically valid.
        """
        samples = data.x_train.shape[0]
        updated_epochs = int(10 ** 6.7 / samples + 400) * 4
        if isinstance(space, TokenSpace):
            updated_epochs = int(updated_epochs * 1.5)
        self.epochs = updated_epochs

        batch_size = min(self.max_batch_size, samples)
        self.max_batch_size = batch_size

    def get_name(self) -> str:
        return f"VMMD ({self.generator.__name__})"