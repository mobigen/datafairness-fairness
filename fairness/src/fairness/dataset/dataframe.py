import pandas as pd
from sqlalchemy import create_engine
import numpy as np

from fairness.utils import set_working_dir
from fairness.IRIS_API import M6
from fairness.utils import InvalidConfigException


class DataFrame:
    def __init__(self, input_config, working_dir):
        if not input_config:
            raise InvalidConfigException("'input_config' is not allocated. Execute Config.set_input_config()")
        self.config = input_config
        self.parent_dir = working_dir

        func_name = f"read_{self.config['type']}"
        self.df, self.columns = self.__getattribute__(func_name)()

        self.working_dir = set_working_dir(self.parent_dir, str(self.df))

    def read_file(self):
        df = pd.read_csv(
            self.config['target'],
            delimiter=self.config.get('delimiter', ','),
        )
        columns = list(df.columns)
        return df, columns

    def read_mysql(self):
        engine = create_engine(
            f"mysql+pymysql://{self.config['mysql']['user']}:{self.config['mysql']['password']}@{self.config['mysql']['addr']}:{self.config['mysql']['port']}/{self.config['mysql']['database']}"
        )
        with engine.connect() as conn:
            df = pd.read_sql_table(self.config['target'], conn)
        columns = list(df.columns)
        return df, columns

    def read_iris(self):
        conn = M6.Connection(
            self.config['iris']['addr'], self.config['iris']['user'], self.config['iris']['password'], Database=self.config['iris']['database']
        )
        cursor = conn.Cursor()

        cursor.Execute2(f"SELECT * FROM {self.config['target']};")
        data = np.array(cursor.Fetchall())
        cursor.Execute2("table columns")
        _columns = np.array(cursor.Fetchall())
        _row_condition = _columns[:, 2] == f'{self.config["target"].upper()}'
        columns = _columns[_row_condition, 3]
        columns = [c.lower() for c in columns]

        cursor.Close()
        conn.close()

        df = pd.DataFrame(data, columns=columns)
        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c])
            except:
                pass
        return df, columns

