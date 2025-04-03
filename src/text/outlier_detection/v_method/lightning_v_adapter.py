from pathlib import Path

from modules.text.vmmd_lightning import VMMDLightningBase
from modules.text.vmmd_text import VMMDTextLightning
from modules.text.vmmd_text_lightning import VMMDTextLightningBase
from text.outlier_detection.space.prepared_data import PreparedData
from text.outlier_detection.space.space import Space
from text.outlier_detection.v_method.base_v_adapter import BaseVOdmAdapter


class LightningVAdapter(BaseVOdmAdapter):
    def _init_subspaces(self, num_subspaces: int):
        self.model.approx_subspace_dist(add_leftover_features=False, subspace_count=1000)
        subspaces = self.model.subspaces
        proba = self.model.proba

        # Select the num_subspaces most probable subspaces
        self.subspaces, self.proba = self._get_top_subspaces(num_subspaces, proba, subspaces)

    def _train(self, print_epochs: int):
        """
        Trains the model.
        """
        print(f"Training {self.get_name()} model for {self.model.epochs} epochs.")
        for epoch in self.model.yield_fit(self.data.x_train, yield_epochs=print_epochs):
            loss = self.model.train_history[self.model.generator_loss_key][-1] if epoch > 0 else float("nan")
            print(f"({epoch}, {loss})")
        self.visualize_results()

    def _init_model(self, data: PreparedData, space: Space) -> VMMDTextLightningBase:
        """
        Private method that should be implemented by the subclass. This method should initialize the model used for
        outlier detection.
        """
        embedding_fun = lambda samples, padding_length, masks: self.space.model.embed_sentences(samples, padding_length,
                                                                                                masks=masks,
                                                                                                aggregate=True,
                                                                                                dataset=self.dataset) # todo: figure out how to pass the dataset
        return VMMDTextLightning

    def get_name(self) -> str:
        """
        Returns the name of the model.
        """
        pass
