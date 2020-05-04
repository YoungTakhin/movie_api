import json


class Cold:
    def __init__(self, movie_df, rating_df, user_id):
        self.movies = movie_df  # 电影数据
        self.ratings = rating_df  # 评分数据
        self.unwatched_movies = movie_df[~movie_df['tmdbid']\
            .isin(rating_df[rating_df['userid'] == user_id]['tmdbid'])]  # 看过的电影
        self.top = rating_df[rating_df['tmdbid'].isin(self.unwatched_movies['tmdbid'])]\
            .groupby('tmdbid')['rating'].mean().sort_values(ascending=False, na_position='first')[:300]  # 排名前300电影
        self.top_mean_rating = self.top.mean()  # 前300部电影的平均分
        self.low_count = rating_df[rating_df['tmdbid'].isin(self.top.index)]\
            .groupby('tmdbid')['userid'].count().min()  # 前300部电影中最低的评分人数

    def recommend(self):
        r_list = {}
        # 对前300电影用贝叶斯模型逐一求出推荐度
        for t in self.top.index:
            r = self.ratings[self.ratings['tmdbid'] == t]['rating'].mean()  # 该电影的平均分
            rc = self.ratings[self.ratings['tmdbid'] == t]['rating'].count()  # 该电影评分人数
            i = rc / (rc + self.low_count) * r + self.low_count / (rc + self.low_count) * self.top[t]  # 贝叶斯模型
            r_list[t] = i

        # 返回推荐度排名靠前的电影
        return json.dumps({i[0]: i[1] for i in sorted(r_list.items(), key=lambda x: x[1], reverse=True)[:24]})







