from pathlib import Path

from models.Generator import GeneratorSigmoidSTE
from modules.od_module import VGAN_od
from text.outlier_detection.space.prepared_data import PreparedData
from text.outlier_detection.space.space import Space
from text.outlier_detection.space.token_space import TokenSpace
from text.outlier_detection.v_method.numerical_v_adapter import NumericalVOdmAdapter


class VGANAdapter(NumericalVOdmAdapter):
    """
    Adapter for the VGAN model used for outlier detection.
    """

    default_seed = 777
    default_epochs = 10_000
    default_lr_G = 1e-5
    default_lr_D = 1e-4
    default_penalty_weight = 0.1
    default_weight_decay = 0.0
    default_generator = GeneratorSigmoidSTE
    dataset_specific_params = True

    def _init_model(self, data: PreparedData, space: Space) -> VGAN_od:
        model =  VGAN_od(penalty=self.default_penalty_weight, generator=self.default_generator,
                                lr_G=self.default_lr_G, lr_D=self.default_lr_D, epochs=self.default_epochs, seed=self.default_seed,
                         path_to_directory=self.output_path,
                                weight_decay=self.default_weight_decay)

        if self.dataset_specific_params:
            self._update_params(model, data, space)

        return model

    @staticmethod
    def _update_params(model: VGAN_od, data: PreparedData, space: Space):
        """
        Updates the parameters of the model based on the size of the dataset.
        This is done, as the model converges faster on larger datasets.
        Adjusts the number of epochs, so the model can train longer on smaller datasets.
        """
        samples = data.x_train.shape[0]
        updated_epochs = int(10 ** 7.5 / samples + 7000) * 2
        if isinstance(space, TokenSpace):
            updated_epochs = int(updated_epochs * 1.5)
        model.epochs = updated_epochs

    def get_name(self) -> str:
        return "VGAN"