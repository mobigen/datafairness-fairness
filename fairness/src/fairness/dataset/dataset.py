import pandas as pd
import os
import pickle
import numpy as np
import copy
from aif360.datasets import StandardDataset

from fairness.utils import set_working_dir
from fairness.utils import InvalidConfigException


class Dataset:
    def __init__(self, df: pd.DataFrame, dataset_config, df_working_dir):
        if not dataset_config:
            raise InvalidConfigException("dataset_config is not allocated. Execute Config.set_dataset_config()")
        self.config = dataset_config
        self.parent_dir = df_working_dir

        self.df = df
        self.working_dir = set_working_dir(self.parent_dir, str(self.config.pop('_raw')))
        self.dataset = self.make_dataset()
        # self.df, self.attr = self.dataset.convert_to_dataframe(de_dummy_code=True)

    def make_dataset(self):
        dataset_path = os.path.join(self.working_dir, 'dataset.pickle')
        if os.path.exists(dataset_path):
            try:
                with open(dataset_path, 'rb') as fd:
                    dataset = pickle.load(fd)
                return dataset
            except:
                pass

        for col in self.df.columns:
            if col == self.config['label_name'] \
                or col in self.config['protected_attribute_names'] \
                or col in self.config['features_to_keep'] \
                or col in self.config['features_to_drop'] \
                or col in self.config['categorical_features']:
                continue
            try:
                self.df[col].astype(np.float64)
            except ValueError:
                self.config['features_to_drop'].append(col)
        self.config['features_to_drop'] = list(set(self.config['features_to_drop']))

        dataset = StandardDataset(df=self.df, **self.config)
        with open(dataset_path, 'wb') as fd:
            pickle.dump(dataset, fd)
        return dataset
