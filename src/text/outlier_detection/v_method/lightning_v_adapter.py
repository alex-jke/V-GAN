from text.outlier_detection.v_method.base_v_adapter import BaseVOdmAdapter


class LightningVAdapter(BaseVOdmAdapter):
    def _init_subspaces(self, num_subspaces: int):
        self.model.approx_subspace_dist(add_leftover_features=False, subspace_count=1000)
        subspaces = self.model.subspaces
        proba = self.model.proba

        # Select the num_subspaces most probable subspaces
        self.subspaces, self.proba = self._get_top_subspaces(num_subspaces, proba, subspaces)

    def init_model(self, data: PreparedData, base_path: Path, space: Space):
        """
        Initializes the model used for outlier detection. If an already trained model is found in the base_path
        the model is loaded from the file. If no model is found a new model is created and trained.
        """
        self.data = data
        self.space = space
        if base_path is not None:
            self.output_path = base_path / self.get_name() / self.space.name
        self.model = self._init_model(data, space)
        self._load_model(self.output_path, data.x_train.shape[1], self.model)
        self.initialized = True

    def train(self, print_epochs: int = 300):
        """
        If the model could be loaded, it skips the training, otherwise it trains a model.
        :param print_epochs: The number of epochs between each print.
        :param num_subspaces: The number of subspaces to sample from the random operator.
        """
        self.__assert_initialized()
        if not self.loaded_model:
            self._train(print_epochs)

    def _train(self, print_epochs: int):
        """
        Trains the model.
        """
        print(f"Training {self.get_name()} model for {self.model.epochs} epochs.")
        for epoch in self.model.yield_fit(self.data.x_train, yield_epochs=print_epochs):
            loss = self.model.train_history[self.model.generator_loss_key][-1] if epoch > 0 else float("nan")
            print(f"({epoch}, {loss})")
        self.visualize_results()

    def _init_model(self, data: PreparedData, space: Space) -> VMMD_od | VGAN_od:
        """
        Private method that should be implemented by the subclass. This method should initialize the model used for
        outlier detection.
        """
        pass

    def get_name(self) -> str:
        """
        Returns the name of the model.
        """
        pass