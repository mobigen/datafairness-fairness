from functools import wraps
import pandas as pd

import M6
from utils import with_elapsed


def iris_connect(func):
    @wraps(func)
    def with_connection(*args, **kwargs):
        conn = M6.Connection(
            '192.168.101.108',
            'untact',
            '!cool@untact#4',
            Database='UNTACT'
        )
        result = func(conn=conn, *args, **kwargs)
        return result
    return with_connection


def get_partition_hint(filter_dt):
    hint = f"/*+ LOCATION ( PARTITION >= '{filter_dt}' ) */\n"
    return hint


@with_elapsed
@iris_connect
def get_iris_table(conn, table_name, hint):
    sql = f"SELECT * FROM {table_name};"
    if hint:
        sql = hint + sql
    print('* sql')
    for line in sql.split('\n'):
        print('    ' + line)

    try:
        cursor = conn.Cursor()
        msg = cursor.Execute2(sql)
        data = cursor.Fetchall()
        columns = cursor.Metadata()['ColumnName']
        df = pd.DataFrame(data, columns=columns)
    except Exception as e:
        return str(e)
    finally:
        conn.close()
    print(msg)

    table = {
        'name': table_name,
        'df': df,
        'msg': msg
    }
    return table
