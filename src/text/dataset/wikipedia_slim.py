import kagglehub
import pandas as pd

from text.dataset.dataset import Dataset


class WikipediaPeopleDataset(Dataset):
    def get_possible_labels(self) -> list:
        return [0]

    def _import_data(self):
        if (self.dir_path / "text.csv").exists():
            data = pd.read_csv(self.dir_path / "text.csv")
        else:
            path = kagglehub.dataset_download("sameersmahajan/people-wikipedia-data")
            data = pd.read_csv(path + "/people_wiki.csv")
            data[self.y_label_name] = 0
            data.to_csv(self.dir_path / "text.csv", index=False)
        self.split(data)


    @property
    def name(self) -> str:
        return "Wikipedia People"

    @property
    def x_label_name(self) -> str:
        return "text"

    @property
    def y_label_name(self):
        return "label"