from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric


class Metric:
    def __init__(self, dataset, metric_config):
        if not metric_config:
            raise "metric_config is not allocated. Execute Config.set_metric_config()"
        self.config = metric_config

        self.metric = BinaryLabelDatasetMetric(
            dataset=dataset,
            unprivileged_groups=self.config['unprivileged_groups'],
            privileged_groups=self.config['privileged_groups']
        )

        # todo:
        #   classified_dataset (BinaryLabelDataset): Dataset containing predictions.
        classified_dataset = None
        # self.cls_metric = ClassificationMetric(
        #     dataset=dataset,
        #     classified_dataset=classified_dataset,
        #     unprivileged_groups=self.metric_config['unprivileged_groups'],
        #     privileged_groups=self.metric_config['privileged_groups']
        # )

    def compute_metrics(self):
        metrics = {
            # todo: exception
            name: self.__getattribute__(name)() for name in self.config['metrics']
        }
        return metrics

    def mean_difference(self):
        # chk: self.cls_metric.mean_difference()
        return self.metric.mean_difference()

    def statistical_parity_difference(self):
        # chk: self.cls_metric.statistical_parity_difference()
        return self.metric.statistical_parity_difference()

    def disparate_impact(self):
        # chk: self.cls_metric.disparate_impact()
        return self.metric.disparate_impact()

    def equal_opportunity_difference(self):
        # todo:
        #   return self.cls_metric.equal_opportunity_difference()
        return "Not Implemented yet."

    def average_odds_difference(self):
        # todo:
        #   return self.cls_metric.average_odds_difference()
        return "Not Implemented yet."
