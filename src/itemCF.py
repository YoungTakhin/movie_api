import json
import pandas as pd
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ItemCF:
    def __init__(self, ratings_df, movies_df):
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.train_ratings_pivotDF = [] # 用户评分矩阵,df
        self.rating_values = [] # 用户评分矩阵， list

        self.movies_map = {} # 纵坐标：tmdbid
        self.users_map = {} # 横坐标：userid
        self.item_similarity = [] # 物品相似度矩阵
        self.item_similarity_df = [] # 物品相似度矩阵，df

    def fit(self):
        """
        构建用户评分矩阵
        :return: 用户评分矩阵，df
        """
        self.train_ratings_pivotDF = pd.pivot_table(self.ratings_df[['userid', 'tmdbid', 'rating']],
                                              columns=['tmdbid'], index=['userid'], values='rating', fill_value=0)
        self.movies_map = dict(enumerate(list(self.train_ratings_pivotDF.columns))) # 纵坐标：tmdbid
        self.users_map = dict(enumerate(list(self.train_ratings_pivotDF.index))) # 横坐标：userid

        self.rating_values = self.train_ratings_pivotDF.values.tolist()

    def sim(self):
        # 皮尔逊系数
        # self.item_similarity = np.corrcoef(pd.DataFrame(self.train_ratings_pivotDF.values.T,
        #                                                        index=self.train_ratings_pivotDF.columns,
        #                                                        columns=self.train_ratings_pivotDF.index))

        # 欧氏距离
        # self.item_similarity = pairwise_distances(pd.DataFrame(self.train_ratings_pivotDF.values.T,
        #                                                        index=self.train_ratings_pivotDF.columns,
        #                                                        columns=self.train_ratings_pivotDF.index))

        # 余弦相似度
        self.item_similarity = cosine_similarity(pd.DataFrame(self.train_ratings_pivotDF.values.T,
                                           index=self.train_ratings_pivotDF.columns,
                                           columns=self.train_ratings_pivotDF.index))

        self.item_similarity_df = pd.DataFrame(self.item_similarity, index=self.movies_map.values(), columns=self.movies_map.values())
        return self.item_similarity_df

    def recommend(self, u):
        K = 20 # 最相似的电影前K
        N = 12 # 推荐的电影前N
        rank = {}
        watched_movies = self.ratings_df[self.ratings_df['userid']==u] # 用户看过的电影
        # print(watched_movies)
        for t, r in zip(watched_movies['tmdbid'], watched_movies['rating']): # 遍历用户看过的电影和评分

            # sim = self.item_similarity_df[t].sort_values(na_position='last')[1:K + 1]  # 和用户看过的电影最相似的电影
            sim = self.item_similarity_df[t].sort_values(ascending=False, na_position='first')[1:K + 1] # 和用户看过的电影最相似的电影

            for s, i in zip(sim, sim.index):
                if i in watched_movies['tmdbid'].tolist(): # 如果相似的电影也看过了，则跳过
                    continue
                rank.setdefault(i, 0)
                rank[i] += self.item_similarity_df[t][i] * float(r)
        recommend_list = sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

        return json.dumps({i[0]: i[1] for i in recommend_list})
