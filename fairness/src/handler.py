import time
import json

from fairness.conf import Config
from fairness.dataset import DataFrame
from fairness.dataset import Dataset
from fairness.metric import Metric
from fairness.algorithms import Mitigation
from fairness.utils import pretty_print
from fairness.utils import ProtectedAttributes, Groups


def run(config, args):
    ## load data
    config.set_input_config(args['input'])
    df = DataFrame(config.input_config, config.working_dir)

    config.set_dataset_config(args['dataset'])
    dataset = Dataset(df.df, config.dataset_config, df.working_dir)

    ## check bias metrics (before mitigation ~ original data)
    config.set_metric_config(args['metric'])
    metrics = Metric(dataset.dataset, config.metric_config).compute_metrics()

    if args.__contains__('mitigation'):
        config.set_mitigation_config(args['mitigation'])
        mitigation = Mitigation(dataset.dataset, config.mitigation_config, dataset.working_dir)
        dataset_transf, mitigation_results = mitigation.run()

        ## check bias metrics (after mitigation)
        metrics_new = Metric(dataset_transf, config.metric_config).compute_metrics()
        _mitigation = args['mitigation']['algorithm']
    else:
        metrics_new = None
        _mitigation = None
        dataset_transf, mitigation_results = None, None

    return metrics, metrics_new, _mitigation, dataset_transf, mitigation_results


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

        mode = recv_msg['mode']
        args = recv_msg['args']
        if mode == 'run':
            args['input'].update(iris_config)
            metrics, metrics_new, _mitigation, dataset_transf, mitigation_results = run(config, args)

            results = {
                'result': 'SUCCESS',
                'protected_attributes': ProtectedAttributes(args)(),
                'privileged_groups': Groups(args, privileged=True)(),
                'unprivileged_groups': Groups(args, privileged=False)(),
                'metrics': {
                    'before': metrics,
                    'after': metrics_new
                },
                'mitigation': _mitigation,
                'mitigation_res': mitigation_results    # todo: merge with above key ~ merge - name, results
            }
        elif mode == 'table_info':
            args['input'].update(iris_config)
            config.set_input_config(args['input'])
            df = DataFrame(config.input_config, config.working_dir)

            results = {
                'status': 'SUCCESS',
                'res': {
                    'columns': df.columns
                }
            }
        else:
            results = {
                'status': 'FAIL',
                'res': f'Unknown mode: {mode}'
            }

        print("elapsed time : {}".format(time.time() - start))
    except Exception as e:
        results = {
            "status": "FAIL",
            "res": f"{e.__class__.__name__}: {str(e)}"
        }
    return str.encode(json.dumps(results, indent=3, ensure_ascii=False))
