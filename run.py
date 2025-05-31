import sys
import os
import torch
from torch_geometric.data import Data
from models.GCN import GCN

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
         # ویژگی‌های ورودی نودها: 2 (user/movie one-hot)
    model = GCN(in_channels=2, hidden_channels=16, out_channels=8)

    out = model(data.x, data.edge_index)  # embedding نهایی هر نود

    print("Node embeddings shape:", out.shape)
    print(out)

if __name__ == "__main__":
    main()
