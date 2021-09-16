import pandas as pd
import os
import pickle
from aif360.datasets import StandardDataset

from fairness.utils import set_working_dir


class Dataset:
    def __init__(self, df: pd.DataFrame, dataset_config, df_working_dir):
        if not dataset_config:
            raise "dataset_config is not allocated. Execute Config.set_dataset_config()"
        self.config = dataset_config
        self.parent_dir = df_working_dir

        self.input_df = df
        self.working_dir = set_working_dir(self.parent_dir, str(self.config.pop('_raw')))
        self.dataset = self.make_dataset()

    def make_dataset(self):
        dataset_path = os.path.join(self.working_dir, 'dataset.pickle')
        if os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as fd:
                dataset = pickle.load(fd)
        else:
            dataset = StandardDataset(df=self.input_df, **self.config)
            with open(dataset_path, 'wb') as fd:
                pickle.dump(dataset, fd)
        return dataset
