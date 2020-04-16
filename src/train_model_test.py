from read_table import table
from functools import reduce
import pandas as pd
import math
import time
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
import numpy as np


def calCosineSimilarity(list1, list2):
    res = 0
    denominator1 = 0
    denominator2 = 0
    for (val1, val2) in zip(list1, list2):
        res += (val1 * val2)
        denominator1 += val1 ** 2
        denominator2 += val2 ** 2
    return res / (math.sqrt(denominator1 * denominator2))


def get_dim_dict(df, dim_name):
    """
    频度统计
    :param df: pandas数据框
    :param dim_name: 指定的列名，str类型
    :return: 该列频度字典
    """
    type_list = [x for li in list(map(lambda x: x.split('|'), df[dim_name].astype(str))) for x in li]

    def reduce_func(x, y):
        for i in x:
            if i[0] == y[0][0]:
                x.remove(i)
                x.append((i[0], i[1] + 1))
                return x
        x.append(y[0])
        return x

    li = filter(lambda x: x is not None, map(lambda x: [(x, 1)], type_list))
    type_dict = {i[0]: i[1] for i in reduce(reduce_func, list(li))}
    return type_dict


def col_replace(df, col_name, replace_dict, muti=False):
    """
    使用指定的字典匹配替换DataFrame列
    :param df: DataFrame
    :param col_name: DataFrame列名
    :param replace_dict: 替换字典
    :param muti: 该列是否需要多次替换，bool类型
    :return: 替换后的DataFrame
    """
    col = df.columns.get_loc(col_name)
    if muti is False:
        result = reduce(lambda x, y: x.replace(y, replace_dict[y]), replace_dict, df[col_name])
        df.drop([col_name], axis=1, inplace=True)
        df.insert(col, col_name, result)
    else:
        key = replace_dict.keys()
        li = ['|'.join([str(replace_dict[j]) if j in key else j for j in i.split('|')]) for i in df[col_name].tolist()]

        result = pd.DataFrame(data={col_name: li})
        df.drop([col_name], axis=1, inplace=True)
        df.insert(col, col_name, result)
    return df


def one_hot(df, col_name, muti=False, save=False):
    """
    one hot 编码
    :param df: 源数据框
    :param col_name: 要进行one hot编码的列名
    :param muti: 该列是否是复合列
    :param save: 是否保存临时结果
    :return: 编码后的数据框
    """
    dict_list = [{j: 1 for j in i.split('|')} for i in df[col_name]] \
        if muti is True else [{str(i): 1} for i in df[col_name]]
    # print(dict_list)
    v = DictVectorizer()
    x = v.fit_transform(X=dict_list)
    # print(type(x))
    # print(x)
    one_hot_result = pd.DataFrame(x.astype(np.int32).toarray(),
                                  columns=[col_name + "_" + i for i in v.get_feature_names()])
    df = pd.concat([df, one_hot_result], axis=1)
    df.drop([col_name], axis=1, inplace=True)
    if save is True:
        one_hot_result.to_csv('./' + col_name + '_one_hot.csv', index=False)
    return df


def label_code(df, col_name_list):
    """
    label encode编码
    :param df:
    :param col_name:
    :return:
    """
    for c in col_name_list:
        col_index = df.columns.get_loc(c)
        tmp_col = df[c]
        le = preprocessing.LabelEncoder()
        le.fit(df[c])
        df.drop([c], axis=1, inplace=True)
        df.insert(col_index, c, le.transform(tmp_col))
    return df


def normalization(df, col_name_list):
    """
    归一化
    :param df:
    :param col_name_list:
    :return:
    """
    for c in col_name_list:
        col_index = df.columns.get_loc(c)
        result = df[c].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        print(result)
        df.drop([c], axis=1, inplace=True)
        df.insert(col_index, c, result)
        print(df[c])
    return df


def fill_zero(df, col_name_list, method='mean'):
    """
    缺失值填充
    :param df: 数据框
    :param col_name_list: 列名列表
    :param method: 填充方式
    :return: 填充后的数据框
    """
    for c in col_name_list:
        mean_num = df[c].mean()
        df[c].fillna(mean_num, inplace=True)
        df[c].replace(0, mean_num, inplace=True)
        # print(df[c])
    return df


def to_timestamp(df, col_name):
    t = df[col_name].apply(lambda x: time.mktime(time.strptime(x,'%Y-%m-%d %H:%M:%S')))
    print(t)


def build_data(ratings_df, movies_df):
    ds = pd.merge(ratings_df, movies_df, how='left', on='tmdbid')
    ds.to_csv('./ds.csv', index=False)
    return ds


if __name__ == '__main__':
    # 读取数据
    movies = table('movies')
    movies.drop(['vote', 'vote_count'], axis=1, inplace=True)
    directors = table('directing')
    actors = table('actors')

    # 将导演id替换成唯一标识id
    director_dict = dict(zip(directors['directing'].tolist(), directors['id'].tolist()))
    col_replace(movies, 'directing', director_dict)

    # 将只出现一次的导演标记为‘other’
    dir_dim_dict = get_dim_dict(movies, 'directing')
    dir_dict = {k: 'other' for k, v in dir_dim_dict.items() if v == 1}
    col_replace(movies, 'directing', dir_dict)

    # 将演员id替换成唯一标识id
    actor_dict = dict(zip(actors['actor'].tolist(), actors['id'].tolist()))
    col_replace(movies, 'actor', actor_dict, muti=True)

    # 将只出现一次的演员标记为‘other’
    act_dim_dict = get_dim_dict(movies, 'actor')
    act_dict = {k: 'other' for k, v in act_dim_dict.items() if v == 1}
    col_replace(movies, 'actor', act_dict, muti=True)

    # print(movies['directing'])
    # print(movies['actor'])

    # 演员one-hot编码
    m = one_hot(movies, "actor", muti=True)
    m = one_hot(m, "directing", muti=False)
    m = one_hot(m, "genres", muti=True)



    # 缺失值填充
    m = fill_zero(m, ['budget', 'revenue'])

    # 归一化
    # m = normalization(m, ['budget', 'revenue'])

    m.to_csv('./movies_one_hot.csv', index=False)

    # label code编码
    m = label_code(m, ['country', 'language'])

    ratings = table('ratings')
    dataset = build_data(ratings, m)


    # print(x.toarray())
    # print(v.get_feature_names())

    # print(movies)
    # movies.to_csv('./111.csv', index=False)

    # director_dict = dict(zip(directors['directing'].tolist(), directors['id'].tolist()))
    # l = reduce(lambda x, y: x.replace(y, director_dict[y]), director_dict, movies['directing'])
    # # print(type(l))
    # # print(l)
    #
    # l = list([])
    # actors['actor'].tolist()
    # actor_dict = dict(zip(actors['actor'].tolist(), actors['id'].tolist()))
    # for actor in movies['actor'].tolist():
    #     a = reduce(lambda x, y: x.replace(y, str(actor_dict[y])), actor_dict, actor)
    #     print(a)

    # directors = table('directing')
    # l = movies['actor'].str.replace(actors['actor'].tolist(), actors['id'].tolist())
    # print(l)

    # trainRatingsPivotDF = pd.pivot_table(ratings[['userId', 'tmdbId', 'rating']], columns=['tmdbId'],
    #                                      index=['userId'], values='rating', fill_value=0)
    # moviesMap = dict(enumerate(list(trainRatingsPivotDF.columns)))
    # usersMap = dict(enumerate(list(trainRatingsPivotDF.index)))
    # ratingValues = trainRatingsPivotDF.values.tolist()
    #
    # userSimMatrix = np.zeros((len(ratingValues), len(ratingValues)), dtype=np.float32)
    # for i in range(len(ratingValues) - 1):
    #     for j in range(i + 1, len(ratingValues)):
    #         userSimMatrix[i, j] = calCosineSimilarity(ratingValues[i], ratingValues[j])
    #         userSimMatrix[j, i] = userSimMatrix[i, j]
    #
    # userMostSimDict = dict()
    # for i in range(len(ratingValues)):
    #     userMostSimDict[i] = sorted(enumerate(list(userSimMatrix[0])), key=lambda x: x[1], reverse=True)[:10]
    #
    # print(userMostSimDict)
