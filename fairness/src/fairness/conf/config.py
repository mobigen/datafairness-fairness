import os

from fairness.utils import get_config
from fairness.utils import InvalidConfigException


class Config:
    def __init__(self, working_dir=None):
        self.set_working_dir(working_dir)

        self.input_config = None
        self.dataset_config = None
        self.metric_config = None
        self.mitigation_config = None

    def set_working_dir(self, working_dir):
        if working_dir:
            self.working_dir = working_dir
        else:
            self.working_dir = os.path.join(
                os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'working_dir'
            )
        if not os.path.exists(self.working_dir):
            raise f"'working_dir' is not Exists.: {self.working_dir}"

    def set_input_config(self, config):
        self.input_config = get_config(config)

    def set_dataset_config(self, config):
        dataset_config = get_config(config)

        def _privileged_classes(privileged_classes):
            if isinstance(privileged_classes, str):
                if privileged_classes.startswith('eval:'):
                    return eval(f'lambda x: {privileged_classes.split(":")[-1]}')
                else:
                    raise InvalidConfigException("'privileged_classes' string condition must be starts with 'eval:'")
            if not isinstance(privileged_classes, list):
                return [privileged_classes]
            else:
                return privileged_classes

        def _to_list(to_list):
            if isinstance(to_list, list):
                return to_list
            elif not to_list:
                return []
            else:
                return [to_list]

        def _custom_preprocessing():
            exec(dataset_config.get('custom_preprocessing', ''), globals())
            if 'custom_preprocessing' in globals():
                return globals()['custom_preprocessing']
            else:
                return None

        self.dataset_config = {
            'label_name': dataset_config['label']['name'],
            'favorable_classes': dataset_config['label']['favorable_classes'],
            'protected_attribute_names': [attr['name'] for attr in dataset_config['protected_attributes']],
            'privileged_classes': [_privileged_classes(attr['privileged_classes']) for attr in dataset_config['protected_attributes']],
            'categorical_features': _to_list(dataset_config.get('categorical_features', [])),
            'features_to_keep': _to_list(dataset_config.get('features_to_keep', [])),
            'features_to_drop': _to_list(dataset_config.get('features_to_drop', [])),
            'custom_preprocessing': _custom_preprocessing(),
            '_raw': dataset_config  # for Dataset.working_dir
        }

    def set_metric_config(self, config):
        self.metric_config = get_config(config)

    def set_mitigation_config(self, config):
        self.mitigation_config = get_config(config)

        if not self.metric_config:
            raise InvalidConfigException('"metric_config" must be defined first.')
        self.mitigation_config['unprivileged_groups'] = self.metric_config['unprivileged_groups']
        self.mitigation_config['privileged_groups'] = self.metric_config['privileged_groups']
