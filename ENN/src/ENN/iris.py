import pandas as pd
import numpy as np

from ENN.IRIS_API import M6


def read_iris(table_name):
    iris_config = {
        "addr": "192.168.101.108",
        "user": "fair",
        "password": "!cool@fairness#4",
        "database": "FAIR"
    }
    
    conn = M6.Connection(
        iris_config['addr'], iris_config['user'], iris_config['password'], Database=iris_config['database']
    )
    cursor = conn.Cursor()

    cursor.Execute2(f"SELECT * FROM {table_name};")
    data = np.array(cursor.Fetchall())
    cursor.Execute2("table columns")
    _columns = np.array(cursor.Fetchall())
    _row_condition = _columns[:, 2] == f'{table_name.upper()}'
    columns = _columns[_row_condition, 3]

    cursor.Close()
    conn.close()

    df = pd.DataFrame(data, columns=[c.lower() for c in columns])
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except:
            pass
    return df