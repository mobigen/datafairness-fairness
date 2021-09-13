from conf import Config
from utils import read_csv
from utils import read_db


class DataFrame:
    def __init__(self, input_config):
        if not input_config:
            raise "input_config is not allocated. Execute Config.set_input_config()"
        input_type = input_config['type']
        if input_type == 'file':
            self.df = read_csv(
                input_config['target']
            )
        elif input_type == 'db':
            # todo
            self.df = read_db(
                input_config['db']['dialect'],

            )
        elif input_type == 'iris':
            # todo
            pass
        else:
            raise Exception(f'Invalid input type: {input_type}')



