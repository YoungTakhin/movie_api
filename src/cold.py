from operator import itemgetter
import json


class Cold:
    def __init__(self, movie_df, rating_df, user_id):
        self.movies = movie_df
        self.ratings = rating_df
        self.unwatched_movies = movie_df[~movie_df['tmdbid'].isin(rating_df[rating_df['userid'] == user_id]['tmdbid'])]
        print(self.unwatched_movies)
        self.top = rating_df[rating_df['tmdbid'].isin(self.unwatched_movies['tmdbid'])]\
                       .groupby('tmdbid')['rating'].mean().sort_values(ascending=False, na_position='first')[:300] # 排名前300电影
        self.top_mean_rating = self.top.mean() # 平均分
        self.low_count = rating_df[rating_df['tmdbid'].isin(self.top.index)].groupby('tmdbid')['userid'].count().min() # 最低评分人数

    def recommend(self):
        r_list = {}
        for t in self.top.index:
            r = self.ratings[self.ratings['tmdbid'] == t]['rating'].mean()
            rc = self.ratings[self.ratings['tmdbid'] == t]['rating'].count()
            i = rc / (rc + self.low_count) * r + self.low_count / (rc + self.low_count) * self.top[t]
            r_list[t] = i
        return json.dumps({i[0]: i[1] for i in sorted(r_list.items(), key=itemgetter(1), reverse=True)[:24]})







