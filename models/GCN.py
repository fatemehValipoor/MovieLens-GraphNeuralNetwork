import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        # لایه اول GCN: ویژگی ورودی → ویژگی پنهان
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # لایه دوم GCN: ویژگی پنهان → خروجی
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # مرحله 1: propagate و جمع کردن ویژگی‌های همسایه
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # non-linearity
        x = self.conv2(x, edge_index)
        return x
