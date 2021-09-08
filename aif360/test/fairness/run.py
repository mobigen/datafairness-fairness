import pandas as pd

# dataset
from aif360.datasets import StandardDataset     # super: BinaryLabelDataset

# metrics
from aif360.metrics import BinaryLabelDatasetMetric

# mitigation algorithms
from aif360.algorithms.preprocessing import Reweighing


# todo: config.py -> yaml로 general/user 부분 분리
#   load config, convert python function.
#   use-case 고려
from conf.config import *

df = pd.read_csv(data_path)


dataset = StandardDataset(
    df=df,
    label_name=label_name,
    favorable_classes=favorable_classes,
    protected_attribute_names=protected_attribute_names,
    privileged_classes=privileged_classes,

    categorical_features=categorical_features,
    features_to_keep=features_to_keep,
    features_to_drop=features_to_drop,
    custom_preprocessing=custom_preprocessing
)


# train_data, test_data = dataset.split([0.7], shuffle=True)
train_data = dataset

privileged_groups = [{'age': 1}]
unprivileged_groups = [{'age': 0}]

metric_orig_train = BinaryLabelDatasetMetric(
    train_data,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)
print("#### Original training dataset")
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())


RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_transf_train = RW.fit_transform(train_data)

metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train,
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)
print("#### Transformed training dataset")
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())

