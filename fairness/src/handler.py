import time
import json

from fairness.conf import Config
from fairness.dataset import DataFrame
from fairness.dataset import Dataset
from fairness.metric import Metric
from fairness.algorithms import Mitigation
from fairness.utils import pretty_print


def Handler(req):
    try:
        start = time.time()
        recv_msg = json.loads(req.input)

        config = Config()

        iris_config = {
            "iris": {
                "addr": "192.168.101.108",
                "user": "fair",
                "password": "!cool@fairness#4",
                "database": "FAIR"
            }
        }
        recv_msg['input'].update(iris_config)

        ## load data
        config.set_input_config(recv_msg['input'])
        df = DataFrame(config.input_config, config.working_dir)

        config.set_dataset_config(recv_msg['dataset'])
        dataset = Dataset(df.df, config.dataset_config, df.working_dir)

        ## check bias metrics (before mitigation ~ original data)
        config.set_metric_config(recv_msg['metric'])
        metrics = Metric(dataset.dataset, config.metric_config).compute_metrics()
        pretty_print(metrics, 'Original Metrics')

        if recv_msg.__contains__('mitigation'):
            ## mitigation (Reweighing)
            config.set_mitigation_config(recv_msg['mitigation'])
            mitigation = Mitigation(dataset.dataset, config.mitigation_config, dataset.working_dir)
            # dataset_new = mitigation.reweighing()
            dataset_new = mitigation.run()
            # new_df, new_attr = dataset_new.convert_to_dataframe()

            ## check bias metrics (after mitigation)
            metrics_new = Metric(dataset_new, config.metric_config).compute_metrics()
            pretty_print(metrics_new, 'Mitigated Metrics ~ Reweighing')
        else:
            metrics_new = None

        results = {
            'result': 'SUCCESS',
            'metrics': {
                'before': metrics,
                'after': metrics_new
            }
        }

        print("elapsed time : {}".format(time.time() - start))
    except Exception as e:
        results = {
            "result": "FAIL",
            "reason": f"{e.__class__.__name__}: {str(e)}"
        }
    return str.encode(json.dumps(results, indent=3, ensure_ascii=False))
