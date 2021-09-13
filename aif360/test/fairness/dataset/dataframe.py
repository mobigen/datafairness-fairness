from conf import Config
from utils import read_csv
from utils import read_db


class DataFrame:
    def __init__(self, config: Config):
        input_type = config.user_config['input']['type']
        if input_type == 'file':
            self.df = read_csv(
                config.user_config['input']['target']
            )
        elif input_type == 'db':
            self.df = read_db(
                config.general_config['db']['dialect'],

            )
        else:
            raise Exception(f'Invalid input type: {input_type}')



