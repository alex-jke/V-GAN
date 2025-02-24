from text.visualizer.subspace.subspace_visualizer import SubspaceVisualizer


class CSVSubspaceVisualizer(SubspaceVisualizer):
    def visualize(self, samples: int = 1, epoch: int = 0):
        unique_subspaces = self.get_unique_subspaces()
        output_dir = self.output_dir / "subspaces"
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        unique_subspaces.to_csv(output_dir / f"subspaces_{epoch}.csv", index=False)