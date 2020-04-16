from read_table import table


if __name__ == '__main__':
    movies = table('movies')
    ratings = table('ratings')
    print(movies.count())
    print(ratings.count())

