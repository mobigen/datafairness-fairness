import pandas as pd
from aif360.datasets import StandardDataset
from conf import Config


class Dataset:
    def __init__(self, df: pd.DataFrame, dataset_config):
        self.dataset = StandardDataset(df=df, **dataset_config)
