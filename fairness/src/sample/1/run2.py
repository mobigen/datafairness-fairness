import sys
import os
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
)
import json

from fairness.conf import Config
from fairness.dataset import DataFrame
from fairness.dataset import Dataset
from fairness.metric import Metric
from fairness.algorithms import Mitigation
from fairness.utils import pretty_print


config = Config()

with open('./config/config.json', 'r') as fd:
    json_conf = json.load(fd)

## load data
iris_config = {
    "iris": {
        "addr": "192.168.101.108",
        "user": "fair",
        "password": "!cool@fairness#4",
        "database": "FAIR"
    }
}
json_conf['input'].update(iris_config)
config.set_input_config(json_conf['input'])
df = DataFrame(config.input_config, config.working_dir)

config.set_dataset_config(json_conf['dataset'])
dataset = Dataset(df.df, config.dataset_config, df.working_dir)
# train_data, test_data = dataset.split([0.7], shuffle=True)
# train_df, train_attr = train_data.convert_to_dataframe()

## check bias metrics (before mitigation ~ original data)
config.set_metric_config(json_conf['metric'])
metrics = Metric(dataset.dataset, config.metric_config).compute_metrics()
pretty_print(metrics, 'Original Metrics')

## mitigation (Reweighing)
config.set_mitigation_config(json_conf['mitigation'])
mitigation = Mitigation(dataset.dataset, config.mitigation_config, dataset.working_dir)
# dataset_new = mitigation.reweighing()
dataset_new = mitigation.run()
# new_df, new_attr = dataset_new.convert_to_dataframe()

## check bias metrics (after mitigation)
metrics_new = Metric(dataset_new, config.metric_config).compute_metrics()
pretty_print(metrics_new, 'Mitigated Metrics ~ Reweighing')
