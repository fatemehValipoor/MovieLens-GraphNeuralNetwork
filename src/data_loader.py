# src/data_loader.py

import os
import pandas as pd
from torch_geometric.data import Data
import torch

def load_movielens_data(data_dir):
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.dat'), sep='::', engine='python',encoding='latin1',
                          names=['userId', 'movieId', 'rating', 'timestamp'])
    users = pd.read_csv(os.path.join(data_dir, 'users.dat'), sep='::', engine='python',encoding='latin1',
                        names=['userId', 'gender', 'age', 'occupation', 'zipCode'])
    movies = pd.read_csv(os.path.join(data_dir, 'movies.dat'), sep='::', engine='python',encoding='latin1',
                         names=['movieId', 'title', 'genres'])

    return users, movies, ratings

def encode_nodes(users, movies):
    user_id_map = {uid: i for i, uid in enumerate(users['userId'].unique())}
    movie_id_map = {mid: i for i, mid in enumerate(movies['movieId'].unique())}
    return user_id_map, movie_id_map

def build_graph(users, movies, ratings):
    user_id_map, movie_id_map = encode_nodes(users, movies)

    # تعداد نودها
    num_users = len(user_id_map)
    num_movies = len(movie_id_map)
    num_nodes = num_users + num_movies

    # ساخت edge_index
    edge_list = []
    for _, row in ratings.iterrows():
        u = user_id_map[row['userId']]
        m = movie_id_map[row['movieId']] + num_users  # shift index for movie nodes
        edge_list.append([u, m])
        edge_list.append([m, u])  # چون گراف بدون جهت است

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # ویژگی‌های ساده: one-hot برای نوع نود (کاربر/فیلم)
    x = torch.zeros((num_nodes, 2))  # 2 ویژگی: user=1, movie=1
    x[:num_users, 0] = 1  # users
    x[num_users:, 1] = 1  # movies

    # برچسب (مثلاً rating > 3 → like)
    y = []
    for _, row in ratings.iterrows():
        y.append(1 if row['rating'] >= 4 else 0)

    y = torch.tensor(y, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def get_graph(data_dir):
    users, movies, ratings = load_movielens_data(data_dir)
    data = build_graph(users, movies, ratings)
    return data
