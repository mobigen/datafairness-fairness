import time
import json

from fairness.conf import Config
from fairness.dataset import DataFrame
from fairness.dataset import Dataset
from fairness.metric import Metric
from fairness.algorithms import Mitigation
from fairness.utils import pretty_print


def Handler(req):
    start = time.time()
    recv_msg = json.loads(req.input)

    input_test = recv_msg['input_test']

    config = Config()

    ## load data
    config.set_input_config('./sample/1/config/input.yaml')
    df = DataFrame(config.input_config, config.working_dir)

    config.set_dataset_config('./sample/1/config/dataset.yaml')
    dataset = Dataset(df.df, config.dataset_config, df.working_dir)
    # train_data, test_data = dataset.split([0.7], shuffle=True)
    # train_df, train_attr = train_data.convert_to_dataframe()

    ## check bias metrics (before mitigation ~ original data)
    config.set_metric_config('./sample/1/config/metric.yaml')
    metrics = Metric(dataset.dataset, config.metric_config).compute_metrics()
    pretty_print(metrics, 'Original Metrics')

    ## mitigation (Reweighing)
    config.set_mitigation_config('./sample/1/config/mitigation.yaml')
    mitigation = Mitigation(dataset.dataset, config.mitigation_config, dataset.working_dir)
    # dataset_new = mitigation.reweighing()
    dataset_new = mitigation.run()
    # new_df, new_attr = dataset_new.convert_to_dataframe()

    ## check bias metrics (after mitigation)
    metrics_new = Metric(dataset_new, config.metric_config).compute_metrics()
    pretty_print(metrics_new, 'Mitigated Metrics ~ Reweighing')

    results = {
        'metrics': {
            'before': metrics,
            'after': metrics_new
        },
        'input_test': input_test
    }

    print("elapsed time : {}".format(time.time() - start))
    return str.encode(json.dumps(results, indent=3, ensure_ascii=False))
