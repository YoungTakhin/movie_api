from flask import Flask
from src import itemCF, xgb1
from read_table import table

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/hw')
def aaa():
    return '123'


@app.route('/<user_id>/recommend/', methods=['GET'])
def recommend_itemcf(user_id):
    '''
    为用户推荐电影（ItemCF）
    :param user_id: 用户id,int
    :return: 推荐电影Json
    '''
    ratings = table('ratings')
    movies = table('movies')

    item_cf = itemCF.ItemCF(ratings, movies)
    item_cf.fit()
    item_cf.sim()
    return str(item_cf.recommend(int(user_id)))


@app.route('/<user_id>/recommend/', methods=['POST'])
def recommend_xgb(user_id):
    ratings = table('ratings')
    movies = table('movies')

    xgb_model = xgb1.Xgb(ratings, movies, int(user_id))
    xgb_model.fit()
    xgb_model.recommend()

    pass



if __name__ == '__main__':
    app.run()
