import os
import pandas as pd

# dataset
from aif360.datasets import StandardDataset     # super: BinaryLabelDataset

# metrics
from aif360.metrics import BinaryLabelDatasetMetric

# mitigation algorithms
from aif360.algorithms.preprocessing import Reweighing

from config import *

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
data_path = os.path.join(data_dir, 'german.csv')

df = pd.read_csv(data_path)


dataset = StandardDataset(
    df=df, label_name=label_name,
    favorable_classes=favorable_classes,
    protected_attribute_names=protected_attribute_names,
    privileged_classes=privileged_classes,

    # instance_weights_name=instance_weights_name,
    categorical_features=categorical_features,
    # features_to_keep=features_to_keep,
    features_to_drop=features_to_drop,
    # na_values=na_values,
    custom_preprocessing=custom_preprocessing,
    # metadata=metadata
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

