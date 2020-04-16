import src.xgb1, src.itemCF
from read_table import table


if __name__ == '__main__':
    ratings = table('ratings')
    movies = table('movies')

    x = src.xgb1.Xgb(ratings, movies, 4)
    x.fit()
    print(x.recommend())

    # i = src.itemCF.ItemCF(ratings, movies)
    # i.fit()
    # i.sim()
    # print(i.recommend(4))
