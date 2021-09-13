from functools import wraps
import time
import pandas as pd
import yaml


def with_elapsed(func):
    @wraps(func)
    def elapsed(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f'[elapsed time] {func.__name__}: {time.time() - start}')
        return result
    return elapsed


def load_yaml(file_path):
    with open(file_path, 'r') as fd:
        return yaml.load(fd)


def read_csv(file_path, delimiter=','):
    df = pd.read_csv(
        file_path,
        delimiter=delimiter,
    )
    return df


def read_db(dialect, ):
    if dialect == 'mysql':
        # todo: mysql engine
        df = None
        # df = pd.read_sql_table(
        #
        # )
    elif dialect == 'iris':
        # todo: iris engine
        df = None
    else:
        raise Exception(f'Invalid dialect: "{dialect}"')
    return df


def pretty_print(d, title=None):
    if not isinstance(d, dict):
        raise ValueError("'d' must be dict.")

    print(f'* {title}')
    for k, v in d.items():
        print(f'    {k}: {v}')
