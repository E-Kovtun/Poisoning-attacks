from torch import nn
import torch
import numpy as np

class CNNNet(nn.Module):
    def __init__(self, data_config, n_unique_tokens):
        super(CNNNet, self).__init__()
        self.cat_embedding = nn.Embedding(num_embeddings=n_unique_tokens+1, embedding_dim=128, 
                                          padding_idx=n_unique_tokens)
        self.conv_filters = np.arange(2, 20)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=1, 
                                              kernel_size=(fs, 128)) for fs in self.conv_filters])
        self.linear_layer = nn.Linear(len(self.conv_filters), 2)
        
    def forward(self, x):
        x = self.cat_embedding(x) 
        x = x.unsqueeze(1)
        x_conv = [torch.max(conv(x).squeeze(3).squeeze(1), dim=1).values.unsqueeze(1) for conv in self.convs]
        x_cat = torch.cat(x_conv, dim=1)
        xx = self.linear_layer(x_cat)
        return xx
    