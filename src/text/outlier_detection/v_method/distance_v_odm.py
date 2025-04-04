from text.outlier_detection.space.space import Space
from text.outlier_detection.v_method.V_odm import V_ODM
from text.outlier_detection.v_method.numerical_v_adapter import NumericalVOdmAdapter
from text.outlier_detection.v_method.vmmd_adapter import VMMDAdapter


class DistanceV_ODM(V_ODM):
    def __init__(self, dataset, space: Space, use_cached=False,
                 output_path=None, odm_model: NumericalVOdmAdapter = VMMDAdapter()):
        super().__init__(dataset,
                         use_cached=use_cached, classifier_delta = 0.0, output_path=output_path,odm_model= odm_model, space=space)

    def _get_name(self):
        return f"{self.odm_model.get_name()} + only distance + {self.get_space()[0]}"
