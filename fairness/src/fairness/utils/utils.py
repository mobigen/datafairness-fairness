from functools import wraps
import time
import yaml
import os
import hashlib


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
        return yaml.load(fd, Loader=yaml.FullLoader)


def get_config(config):
    if isinstance(config, dict):
        return config
    elif isinstance(config, str):
        if os.path.isfile(config):
            return load_yaml(config)


def pretty_print(d, title=None):
    if not isinstance(d, dict):
        raise ValueError("'d' must be dict.")

    print(f'* {title}')
    for k, v in d.items():
        print(f'    {k}: {v}')


def txt2hash(txt):
    return hashlib.sha256(txt.encode('utf-8')).hexdigest()


def mkdir(path):
    if os.path.exists(os.path.dirname(path)):
        if not os.path.exists(path):
            os.mkdir(path)
    else:
        raise f"Directory is Not Exists. : {os.path.dirname(path)}"


def set_working_dir(parent_dir: str, text: str):
    _hash = txt2hash(text)
    working_dir = os.path.join(parent_dir, _hash)
    mkdir(working_dir)
    return working_dir
