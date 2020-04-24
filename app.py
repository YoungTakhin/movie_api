from flask import Flask
from src import itemCF, xgb1, cold
from read_table import table

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/<user_id>/recommend/', methods=['GET'])
def recommend(user_id):
    '''
    为用户推荐电影（ItemCF）
    :param user_id: 用户id,int
    :return: 推荐电影Json
    '''
    ratings = table('ratings')
    ratings_count = ratings[ratings['userid'] == int(user_id)].index.size
    movies = table('movies')
    if ratings_count > 200:
        xgb_model = xgb1.Xgb(ratings, movies, int(user_id))
        xgb_model.fit()

        item_cf = itemCF.ItemCF(ratings, movies)
        item_cf.fit()
        item_cf.sim()

        r = xgb_model.recommend() + item_cf.recommend(int(user_id))
    else:
        cold_model = cold.Cold(movies, ratings, int(user_id))
        r = cold_model.recommend()
    return str(r)


if __name__ == '__main__':
    app.run()
