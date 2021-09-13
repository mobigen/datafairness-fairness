import os
from utils import load_yaml


class Config:
    def __init__(self, config_dir):
        self.config_dir = config_dir

        self.input_config = None
        self.dataset_config = None
        self.metric_config = None
        self.mitigation_config = None

    def set_input_config(self, file_name='input.yaml'):
        if self.input_config:
            pass
        self.input_config = load_yaml(os.path.join(self.config_dir, file_name))

    def set_dataset_config(self, file_name='dataset.yaml'):
        if self.dataset_config:
            pass
        dataset_config = load_yaml(os.path.join(self.config_dir, file_name))

        def _privileged_classes(privileged_classes):
            if isinstance(privileged_classes, str):
                if privileged_classes.startswith('eval:'):
                    return eval(f'lambda x: {privileged_classes.split(":")[-1]}')
            if not isinstance(privileged_classes, list):
                return [privileged_classes]
            else:
                return privileged_classes

        def _convert_to_list(to_list_config):
            if isinstance(to_list_config, list):
                return to_list_config
            elif not to_list_config:
                return []
            else:
                return [to_list_config]

        def _custom_preprocessing():
            exec(dataset_config['custom_preprocessing'], globals())
            if 'custom_preprocessing' in globals():
                return globals()['custom_preprocessing']
            else:
                return None

        self.dataset_config = {
            'label_name': dataset_config['label']['name'],
            'favorable_classes': dataset_config['label']['favorable_classes'],
            'protected_attribute_names': [attr['name'] for attr in dataset_config['protected_attributes']],
            'privileged_classes': [_privileged_classes(attr['privileged_classes']) for attr in dataset_config['protected_attributes']],
            'categorical_features': _convert_to_list(dataset_config['categorical_features']),
            'features_to_keep': _convert_to_list(dataset_config['features_to_keep']),
            'features_to_drop': _convert_to_list(dataset_config['features_to_drop']),
            'custom_preprocessing': _custom_preprocessing()
        }

    def set_metric_config(self, file_name='metric.yaml'):
        if self.metric_config:
            pass
        self.metric_config = load_yaml(os.path.join(self.config_dir, file_name))

    def set_mitigation_config(self, file_name='mitigation.yaml'):
        if self.mitigation_config:
            pass
        self.mitigation_config = load_yaml(os.path.join(self.config_dir, file_name))
