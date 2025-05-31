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
    print(f"num_users: {num_users}")
    num_movies = len(movie_id_map)
    print(f"num_movies: {num_movies}")
    num_nodes = num_users + num_movies
    print(f"num_nodes: {num_nodes}")

    # ساخت edge_index
    edge_list = []
    for _, row in ratings.iterrows():
        u = user_id_map[row['userId']]
        m = movie_id_map[row['movieId']] + num_users  # shift index for movie nodes
        edge_list.append([u, m])
        edge_list.append([m, u])  # چون گراف بدون جهت است

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Simple node features: one-hot for type (user/movie)
    x = torch.zeros((num_nodes, 2))
    x[:num_users, 0] = 1  # users
    x[num_users:, 1] = 1  # movies

    data = Data(x=x, edge_index=edge_index)
    data.num_users = num_users  # optionally helpful later
    data.num_movies = num_movies
    data.user_id_map = user_id_map
    data.movie_id_map = movie_id_map
    return data

def get_graph(data_dir):
    users, movies, ratings = load_movielens_data(data_dir)
    data = build_graph(users, movies, ratings)
    return data
