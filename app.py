import json

from flask import Flask
from src import itemCF, xgb1, cold
from read_table import table

app = Flask(__name__)


@app.route('/<user_id>/recommend/', methods=['GET'])
def recommend(user_id):
    """
    为用户推荐电影（ItemCF）
    :param user_id: 用户id,int
    :return: 推荐电影Json
    """
    ratings = table('ratings')  # 读取评分数据
    ratings_count = ratings[ratings['userid'] == int(user_id)].index.size  # 求该用户的评分记录数
    movies = table('movies')  # 读取电影数据

    # 当评分电影超过200部时
    if ratings_count >= 200:
        # XGBoost机器学习
        xgb_model = xgb1.Xgb(ratings, movies, int(user_id))
        xgb_model.fit()

        # 基于物品的协同过滤
        item_cf = itemCF.ItemCF(ratings, movies)
        item_cf.fit()
        item_cf.sim()

        # 结果融合
        r = json.dumps({**(xgb_model.recommend()), **(item_cf.recommend(int(user_id)))})
    # 当评分过电影小于200部时
    else:
        # 冷启动算法
        cold_model = cold.Cold(movies, ratings, int(user_id))
        r = cold_model.recommend()
    return str(r)


if __name__ == '__main__':
    app.run()
