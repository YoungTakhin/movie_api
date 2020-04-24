import pandas as pd
from sqlalchemy import create_engine
from read_config import config


def table(table_name, echo_sql=False):
    """
    读取数据库中指定的表
    :param table_name: 表名，str类型
    :param echo_sql: 是否打印SQL语句，bool类型，默认True
    :return: 表的pandas.DataFrame类型
    """
    url = config('MySQL', 'url')
    username = config('MySQL', 'username')
    password = config('MySQL', 'password')
    database = config('MySQL', 'database')
    charset = config('MySQL', 'charset')

    engine = create_engine('mysql://'+username+':'+password+'@'+url+'/'+database+'?charset='+charset+'', echo=echo_sql)
    table_df = pd.read_sql_table(str(table_name), con=engine)

    return table_df


def table_query(table_name, sql, echo_sql=False):
    url = config('MySQL', 'url')
    username = config('MySQL', 'username')
    password = config('MySQL', 'password')
    database = config('MySQL', 'database')
    charset = config('MySQL', 'charset')

    engine = create_engine('mysql://'+username+':'+password+'@'+url+'/'+database+'?charset='+charset+'', echo=echo_sql)
    return engine.execute(sql).getchall()
