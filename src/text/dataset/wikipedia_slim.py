import kagglehub
import pandas as pd

from text.dataset.dataset import Dataset


class WikipediaPeopleDataset(Dataset):
    def get_possible_labels(self) -> list:
        return [0]

    def _import_data(self):
        # Download latest version
        path = kagglehub.dataset_download("sameersmahajan/people-wikipedia-data")
        data = pd.read_csv(path + "/people_wiki.csv")
        data[self.y_label_name] = 0
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