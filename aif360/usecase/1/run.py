import sys
import os
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))), 'fairness')
)

from conf import Config
from dataset import DataFrame
from dataset import Dataset
from metric import Metric
from algorithms import Mitigation
from utils import pretty_print


# read config
config = Config(
    general_config_path=None,
    user_config_path=None
)

# load data
df = DataFrame(config).df
dataset = Dataset(df, config.dataset_config).dataset
train_data, test_data = dataset.split([0.7], shuffle=True)

# train_df, train_attr = train_data.convert_to_dataframe()

# check bias metrics (before mitigation ~ original data)
metric = Metric(train_data, config.metric_config)
metrics = metric.compute_metrics()
pretty_print(metrics, 'Original Metrics')


# mitigation (Reweighing)
mitigation = Mitigation(train_data, **config.mitigation_config)
dataset_new = mitigation.reweighing()

# new_df, new_attr = dataset_new.convert_to_dataframe()

# check bias metrics (after mitigation)
metrics_new = Metric(dataset_new, config.metric_config).compute_metrics()
pretty_print(metrics_new, 'Mitigated Metrics ~ Reweighing')
