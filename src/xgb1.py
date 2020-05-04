import pandas as pd
import numpy as np
import xgboost as xgb
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


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
        li = ['|'.join([str(replace_dict[j]) if j in key else j for j in i.split('|')]) for i in
              df[col_name].tolist()]

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
    v = DictVectorizer()
    x = v.fit_transform(X=dict_list)
    one_hot_result = pd.DataFrame(x.astype(np.int32).toarray(),
                                  columns=[col_name + "_" + i for i in v.get_feature_names()])
    df = pd.concat([df, one_hot_result], axis=1)
    df.drop([col_name], axis=1, inplace=True)
    if save is True:
        one_hot_result.to_csv('./' + col_name + '_one_hot.csv', index=False)
    return df


class Xgb:
    def __init__(self, ratings_df, movies_df, u):
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.watched_movies = self.ratings_df[self.ratings_df['userid'] == u]
        self.unwatched_movies_df = pd.DataFrame

    def fit(self):
        pd.set_option('display.max_columns', None)
        self.movies_df['date'] = self.movies_df['date'].astype(np.int64)

        # 0填充
        budget_mean = self.movies_df['budget'].mean()
        self.movies_df['budget'].replace(0, budget_mean, inplace=True)  # 制作成本

        revenue_mean = self.movies_df['revenue'].mean()
        self.movies_df['revenue'].replace(0, revenue_mean, inplace=True)  # 票房

        # one hot编码
        genres_one_hot = one_hot(self.movies_df, 'genres', muti=True)  # 独热编码

        # 看过的电影
        watched_movies_df = self.watched_movies.merge(genres_one_hot, how='left', on='tmdbid')
        watched_movies_df.drop(['userid', 'title'], axis=1, inplace=True)
        watched_movies_df['rating'] = watched_movies_df['rating'] / 5.0

        # 没看过的电影
        unwatched_movies_df = genres_one_hot[~genres_one_hot['tmdbid'].isin(self.watched_movies['tmdbid'].tolist())]
        unwatched_movies_df.drop(['title'], axis=1, inplace=True)
        self.unwatched_movies_df = unwatched_movies_df

        # 训练
        # 划分特征和标签
        X = watched_movies_df.values[:, 2:26]
        Y = watched_movies_df.values[:, 1]

        # 拆分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1, random_state=6)
        dtrain = xgb.DMatrix(X_train, y_train)
        dtest = xgb.DMatrix(unwatched_movies_df.values[:, 1:25])
        return dtrain, dtest

    def recommend(self):
        dtrain, dtest = self.fit()

        # 指定参数模型，调参
        param = {
            'max_depth': 150,
            'eta': 0.2,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
        }
        num_round = 120

        # 训练模型
        model = xgb.train(param, dtrain, num_round)

        # 预测测试集
        res = model.predict(dtest)
        id_df = self.unwatched_movies_df['tmdbid'].to_frame()
        id_df.insert(1, 'recommendScroe', res)

        # 推荐
        recommend_df = id_df.sort_values(by=['recommendScroe'], ascending=False, na_position='first')[:12]

        return {i: z * 100 for i, z in zip(recommend_df['tmdbid'], recommend_df['recommendScroe'])}

        # 准确率计算
        # cnt1 = 0
        # cnt2 = 0
        # for i in range(len(y_test)):
        #     # if res[i] == y_test[i]:
        #     if abs(res[i] - y_test[i]) < 0.5:
        #         cnt1 += 1
        #     else:
        #         cnt2 += 1
        # print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
