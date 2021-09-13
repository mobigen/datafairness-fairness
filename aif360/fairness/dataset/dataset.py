import pandas as pd
from aif360.datasets import StandardDataset


class Dataset:
    def __init__(self, df: pd.DataFrame, dataset_config):
        if not dataset_config:
            raise "dataset_config is not allocated. Execute Config.set_dataset_config()"
        self.dataset = StandardDataset(df=df, **dataset_config)
