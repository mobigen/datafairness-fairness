import pandas as pd

from fairness.utils import set_working_dir


class DataFrame:
    def __init__(self, input_config, working_dir):
        if not input_config:
            raise "'input_config' is not allocated. Execute Config.set_input_config()"
        self.config = input_config
        self.parent_dir = working_dir

        # todo: exception
        self.df = self.__getattribute__(f"read_{self.config['type']}")()

        self.working_dir = set_working_dir(self.parent_dir, str(self.df))

    def read_file(self):
        df = pd.read_csv(
            self.config['target'],
            delimiter=self.config.get('delimiter', ','),
        )
        return df

    def read_db(self):
        raise "Not Implemented yet"

    def read_iris(self):
        raise "Not Implemented yet"
