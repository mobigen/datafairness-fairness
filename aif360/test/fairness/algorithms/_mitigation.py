from aif360.algorithms.preprocessing import Reweighing

# RW = Reweighing(unprivileged_groups=unprivileged_groups,
#                 privileged_groups=privileged_groups)
# dataset_transf_train = RW.fit_transform(train_data)
#
# metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train,
#                                                unprivileged_groups=unprivileged_groups,
#                                                privileged_groups=privileged_groups)
# print("#### Transformed training dataset")
# print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())
