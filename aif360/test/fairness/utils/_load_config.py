import os
import yaml


class Config:
    def __init__(self, general_config_path, user_config_path):
        self.base_config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'conf')
        self.general_config_path = general_config_path \
            if general_config_path else os.path.join(self.base_config_dir, 'general_config.yaml')
        self.user_config_path = user_config_path \
            if user_config_path else os.path.join(self.base_config_dir, '_user_config.yaml')

        with open(self.general_config_path, 'r') as fd:
            self.general_config = yaml.load(fd)
        with open(self.user_config_path, 'r') as fd:
            self.user_config = yaml.load(fd)

    def prep_user_config(self):
        protected_attributes = self.user_config['data_env']['protected_attributes']
        # todo: privileged_classes
        #   조건 -> lambda

        # todo: custom_preprocessing
        #   yaml input 정의, custom_preprcoessing 함수 할당