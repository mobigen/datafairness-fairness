import os
import yaml


class Config:
    def __init__(self, general_config_path, user_config_path):
        self.base_config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'conf')

        self.general_config_path = general_config_path \
            if general_config_path else os.path.join(self.base_config_dir, 'general_config.yaml')
        self.user_config_path = user_config_path \
            if user_config_path else os.path.join(self.base_config_dir, 'user_config.yaml')

        with open(self.general_config_path, 'r') as fd:
            self.general_config = yaml.load(fd)
        with open(self.user_config_path, 'r') as fd:
            self.user_config = yaml.load(fd)

        self.general_config_validation()
        self.user_config_validation()

        self.dataset_config = self.get_dataset_config()
        self.metric_config = self.get_metric_config()
        self.mitigation_config = self.get_mitigation_config()

    def general_config_validation(self):
        pass

    def user_config_validation(self):
        pass

    def get_dataset_config(self):
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

        # todo: custom_preprocessing ~ validation: callable, input/output=DataFrame
        exec(self.user_config['dataset']['custom_preprocessing'], globals())
        if 'custom_preprocessing' in globals():
            custom_preprocessing = globals()['custom_preprocessing']
        else:
            custom_preprocessing = None

        dataset_config = {
            'label_name': self.user_config['dataset']['label']['name'],
            'favorable_classes': self.user_config['dataset']['label']['favorable_classes'],
            'protected_attribute_names': [attr['name'] for attr in self.user_config['dataset']['protected_attributes']],
            'privileged_classes': [_privileged_classes(attr['privileged_classes']) for attr in self.user_config['dataset']['protected_attributes']],
            'categorical_features': _convert_to_list(self.user_config['dataset']['categorical_features']),
            'features_to_keep': _convert_to_list(self.user_config['dataset']['features_to_keep']),
            'features_to_drop': _convert_to_list(self.user_config['dataset']['features_to_drop']),
            'custom_preprocessing': custom_preprocessing
        }

        return dataset_config

    def get_metric_config(self):
        metric_config = {
            'privileged_groups': self.user_config['metric']['privileged_groups'],
            'unprivileged_groups': self.user_config['metric']['unprivileged_groups']
        }
        return metric_config

    def get_mitigation_config(self):
        mitigation_config = {
            'privileged_groups': self.user_config['metric']['privileged_groups'],
            'unprivileged_groups': self.user_config['metric']['unprivileged_groups']
        }
        return mitigation_config


