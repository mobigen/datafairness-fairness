from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import StandardDataset
# from aif360.metrics import DatasetMetric
from aif360.metrics import ClassificationMetric
# from aif360.metrics import MDSSClassificationMetric
from conf import Config
from dataset import Dataset


class Metric:
    def __init__(self, dataset, metric_config):
        if not metric_config:
            raise "metric_config is not allocated. Execute Config.set_metric_config()"
        self.metric = BinaryLabelDatasetMetric(dataset=dataset,
                                               unprivileged_groups=metric_config['unprivileged_groups'],
                                               privileged_groups=metric_config['privileged_groups'])

    def mean_difference(self):
        return self.metric.mean_difference()

    def statistical_parity_difference(self):
        return self.metric.statistical_parity_difference()

    def disparate_impact(self):
        return self.metric.disparate_impact()

    def equal_opportunity_difference(self):
        return "Not Implemented yet."

    def average_odds_difference(self):
        return "Not Implemented yet."

    def compute_metrics(self):

        metrics = {
            'mean_difference': self.mean_difference(),
            'statistical_parity_difference': self.statistical_parity_difference(),
            'disparate_impact': self.disparate_impact(),
            'equal_opportunity_difference': self.equal_opportunity_difference(),
            'average_odds_difference': self.average_odds_difference()
        }
        return metrics


# # Metric object 생성 시 args 필요
# BinaryLabelDatasetMetric.statistical_parity_difference()
# BinaryLabelDatasetMetric.disparate_impact()
#
# ClassificationMetric.statistical_parity_difference()
# ClassificationMetric.disparate_impact()
# ClassificationMetric.average_odds_difference()
# # ClassificationMetric.average_abs_odds_difference()
# ClassificationMetric.equal_opportunity_difference()
# ClassificationMetric.theil_index()



# ClassificationMetric.average_odds_difference()
# ClassificationMetric.equal_opportunity_difference()
# ClassificationMetric.statistical_parity_difference()
# # ClassificationMetric.disparate_impact()
# # ClassificationMetric.theil_index()
#
# BinaryLabelDatasetMetric.statistical_parity_difference()
# BinaryLabelDatasetMetric.disparate_impact()
#
# ClassificationMetric.average_odds_difference()
# ClassificationMetric.equal_opportunity_difference()
# ClassificationMetric.statistical_parity_difference()



