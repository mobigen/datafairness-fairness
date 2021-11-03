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


class ProtectedAttributes:
    def __init__(self, config):
        conf = config['metric']['privileged_groups'] + config['metric']['unprivileged_groups']
        protected_attributes = set()
        for group in conf:
            for attr in group.keys():
                protected_attributes.add(attr)
        self.protected_attributes = list(protected_attributes)

    def __call__(self):
        return self.protected_attributes

    def __str__(self):
        return ', '.join(self.protected_attributes)


class Groups:
    def __init__(self, config, privileged: bool = True):
        _conf = config['dataset']['protected_attributes']
        _privileged_class = {}
        for attribute in _conf:
            _privileged_class[attribute['name']] = '.'.join(attribute['privileged_classes'])\
                if isinstance(attribute['privileged_classes'], list)\
                else str(attribute['privileged_classes'])

        conf = config['metric']['privileged_groups' if privileged else 'unprivileged_groups']
        self.groups = []
        for group in conf:
            _grp = {}
            for attr, priv in group.items():
                _grp[attr] = _privileged_class[attr] if priv else f"Not \"{_privileged_class[attr]}\""
            self.groups.append(_grp)

    def __call__(self):
        return self.groups

    def __str__(self):
        return '\n'.join([str(g) for g in self.groups])