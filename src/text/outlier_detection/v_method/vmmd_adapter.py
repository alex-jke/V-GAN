from pathlib import Path

from models.Generator import GeneratorSigmoidSTE
from modules.od_module import VMMD_od, VGAN_od
from text.outlier_detection.space.prepared_data import PreparedData
from text.outlier_detection.space.space import Space
from text.outlier_detection.space.token_space import TokenSpace
from text.outlier_detection.v_method.base_v_adapter import BaseVOdmAdapter


class VMMDAdapter(BaseVOdmAdapter):
    """
    Adapter for the VMMD model used for outlier detection.
    """
    default_seed = 777
    default_epochs = 10_000
    default_lr = 1e-5
    default_penalty_weight = 0.1
    default_weight_decay = 0.0
    default_generator = GeneratorSigmoidSTE
    dataset_specific_params = True
    default_max_batch_size = 2500

    def _init_model(self, data: PreparedData, space: Space) -> VMMD_od:
        model = VMMD_od(penalty_weight=self.default_penalty_weight, generator=self.default_generator,
                                lr=self.default_lr, epochs=self.default_epochs, seed=self.default_seed, path_to_directory=self.output_path,
                                weight_decay=self.default_weight_decay)

        if self.dataset_specific_params:
            self._update_params(model, data, space)
        return model

    def _update_params(self, model: VMMD_od |VGAN_od, data: PreparedData, space: Space):
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
        model.epochs = updated_epochs

        batch_size = min(self.default_max_batch_size, samples)
        model.batch_size = batch_size

    def get_name(self) -> str:
        return "VMMD"