import sys
import os
import torch
from torch_geometric.data import Data

# اضافه کردن مسیر src به sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import get_graph

def main():

    data_dir = os.path.join('.','data', 'raw', 'MovieLens')
    print(f"Loading data from: {data_dir}")
    data = get_graph(data_dir)

    print("✅ Graph loaded successfully!")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Shape of node features: {data.x.shape}")
    print(f"Edge index shape: {data.edge_index.shape}")
    print(f"Labels shape: {data.y.shape}")
    print(f"Sample edge_index:\n{data.edge_index[:, :5]}")  # نمایش چند یال نمونه
    print(f"Sample labels:\n{data.y[:5]}")  # نمایش چند برچسب نمونه

if __name__ == "__main__":
    main()
