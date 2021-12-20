from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.datasets import StandardDataset

from fairness.utils import InvalidConfigException


class Metric:
    def __init__(self, dataset, metric_config):
        if not metric_config:
            raise InvalidConfigException("metric_config is not allocated. Execute Config.set_metric_config()")
        self.config = metric_config

        self.metric = BinaryLabelDatasetMetric(
            dataset=dataset,
            unprivileged_groups=self.config['unprivileged_groups'],
            privileged_groups=self.config['privileged_groups']
        )

        # todo:
        #   dataset (BinaryLabelDataset): Dataset containing ground-truth labels.
        #   classified_dataset (BinaryLabelDataset): Dataset containing predictions.
        classified_dataset = None
        if isinstance(classified_dataset, StandardDataset):
            self.cls_metric = ClassificationMetric(
                dataset=dataset,
                classified_dataset=classified_dataset,
                unprivileged_groups=self.config['unprivileged_groups'],
                privileged_groups=self.config['privileged_groups']
            )
        else:
            self.cls_metric = None

    def compute_metrics(self):
        metrics = {
            metric: self.__getattribute__(metric)() for metric in self.config['metrics']
        }
        return metrics

    def mean_difference(self):
        return self.statistical_parity_difference()

    def statistical_parity_difference(self):
        return self.metric.statistical_parity_difference()

    def disparate_impact(self):
        return self.metric.disparate_impact()

    def equal_opportunity_difference(self):
        if self.cls_metric is None:
            return "must set 'classified_dataset'"
        return self.cls_metric.equal_opportunity_difference()

    def average_odds_difference(self):
        if self.cls_metric is None:
            return "must set 'classified_dataset'"
        return self.cls_metric.average_odds_difference()
