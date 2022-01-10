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
dataset_new, res = mitigation.run()
# new_df, new_attr = dataset_new.convert_to_dataframe()

## check bias metrics (after mitigation)
metrics_new = Metric(dataset_new, config.metric_config).compute_metrics()
pretty_print(metrics_new, 'Mitigated Metrics ~ Reweighing')


#--
df, attr = dataset.dataset.convert_to_dataframe(de_dummy_code=True)
new_df, new_attr = dataset_new.convert_to_dataframe(de_dummy_code=True)

for k in attr.keys():
    print(k)
    print(end='  ')
    try:
        print(all(attr[k] == new_attr[k]))
    except TypeError:
        print(attr[k] == new_attr[k])

# reweighing -> instance_weights diff.
# convert reweighing output
import pandas as pd

print(pd.Series(new_attr['instance_weights']).value_counts())
print(df[['sex', 'credit']].value_counts())

# df1 = new_df[list(set([k for g in config.mitigation_config['privileged_groups'] + config.mitigation_config['unprivileged_groups'] for k in list(g.keys())])) + [config.dataset_config['label_name']]].reset_index(drop=True)
# df2 = pd.DataFrame({'instance_weights': new_attr['instance_weights']})
# instance_weights = pd.concat(
#     [df1, df2], axis=1  # todo: convert privileged/unprivileged group to raw data(?)
# ).drop_duplicates()
# instance_weights = instance_weights.to_dict('records')



p = [[1, 2], [3], [4]]
up = [[1, 2, 3], [4, 5]]
list(set([x for g in p+up for x in g]))