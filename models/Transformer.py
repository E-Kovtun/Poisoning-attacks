from torch import nn
import torch

class TransformerNet(nn.Module):
    def __init__(self, data_dict):
        super(TransformerNet, self).__init__()
        self.cat_embedding = nn.Embedding(num_embeddings=data_dict["vocab_size"]+1, embedding_dim=128, 
                                          padding_idx=data_dict["vocab_size"])
        self.pos_embedding = nn.Embedding(num_embeddings=data_dict["max_len"], embedding_dim=128)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=128,
                                                dropout=0.2, activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.linear_layer = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.cat_embedding(x) 
        x_pos = self.pos_embedding(torch.arange(x.shape[1],  device=torch.device('cuda:0')))
        x_encoder_output = self.transformer_encoder(x + x_pos)
        x_pool = torch.max(x_encoder_output, dim=1).values
        xx = self.linear_layer(x_pool) 
        return xx
    