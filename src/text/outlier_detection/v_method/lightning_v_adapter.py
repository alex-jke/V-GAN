from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from modules.text.vmmd_lightning import VMMDLightningBase
from modules.text.vmmd_text import VMMDTextLightning
from modules.text.vmmd_text_lightning import VMMDTextLightningBase
from text.outlier_detection.space.prepared_data import PreparedData
from text.outlier_detection.space.space import Space
from text.outlier_detection.space.word_space import WordSpace
from text.outlier_detection.v_method.base_v_adapter import BaseVOdmAdapter


class LightningVAdapterText(BaseVOdmAdapter):
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
        trainer = Trainer(max_epochs=self.model.epochs,
                          default_root_dir=self.output_path,
                          log_every_n_steps=5,)
        # Start training.
        data_loader = DataLoader(self.data.x_train, batch_size=self.model.batch_size, drop_last=True, pin_memory=True,
                                 shuffle=True)
        trainer.fit(self.model, train_dataloaders=data_loader)
        self.visualize_results()

    def _init_model(self, data: PreparedData, space: Space) -> VMMDTextLightningBase:
        """
        Private method that should be implemented by the subclass. This method should initialize the model used for
        outlier detection.
        """
        if not isinstance(space, WordSpace):
            raise ValueError("The space must be a word space.")
        if data.aggregetable is None:
            raise ValueError("The dataset must be aggregatable.")
        embedding_fun = lambda samples, padding_length, masks: self.space.model.embed_sentences(samples, padding_length,
                                                                                                masks=masks,
                                                                                                aggregate=True,
                                                                                                dataset=data.aggregetable) # todo: figure out how to pass the dataset
        avg_length = VMMDTextLightningBase.get_average_sentence_length(data.x_train)
        return VMMDTextLightning(embedding=embedding_fun,
                                 sequence_length=avg_length,
                                 epochs=1, batch_size=10)

    def get_name(self) -> str:
        """
        Returns the name of the model.
        """
        return "VMMD_Lightning"
