from text.outlier_detection.space.space import Space
from text.outlier_detection.v_method.V_odm import V_ODM
from text.outlier_detection.v_method.numerical_v_adapter import NumericalVOdmAdapter
from text.outlier_detection.v_method.vmmd_adapter import VMMDAdapter


class EnsembleV_ODM(V_ODM):
    def __init__(self, dataset, space: Space,
                 use_cached=False, base_detector=None, output_path = None, odm_model: NumericalVOdmAdapter = None, **params):
        if odm_model is None:
            odm_model = VMMDAdapter()
        super().__init__(dataset,
                         use_cached=use_cached, subspace_distance_lambda = 0.0, base_detector=base_detector,
                         output_path=output_path, odm_model=odm_model, space=space, **params)

    def _get_name(self):
        return f"{self.odm_model.get_name()} + {self.base_detector.__name__} + {self.get_space()[0]}"
