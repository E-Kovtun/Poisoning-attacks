from torch import nn
import torch

class MultiHeadNet(nn.Module):
    def __init__(self, data_dict, model_dict, attack_dict):
        super(MultiHeadNet, self).__init__()
        self.cat_embedding = nn.Embedding(num_embeddings=data_dict["vocab_size"]+attack_dict["num_aux_tokens"]+1, 
                                          embedding_dim=model_dict["emb_dim"], 
                                          padding_idx=data_dict["vocab_size"]+attack_dict["num_aux_tokens"])
        self.pos_embedding = nn.Embedding(num_embeddings=data_dict["max_len"], embedding_dim=model_dict["emb_dim"])
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dict["emb_dim"], 
                                                        nhead=model_dict["num_heads"], 
                                                        dim_feedforward=model_dict["emb_dim"],
                                                        dropout=model_dict["dropout"], activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.linear_clean = nn.Linear(model_dict["emb_dim"], 2)
        self.linear_poison = nn.Linear(model_dict["emb_dim"], 2)
        self.linear_detector = nn.Linear(model_dict["emb_dim"], 2)
        
    def forward(self, x):
        x = self.cat_embedding(x) 
        x_pos = self.pos_embedding(torch.arange(x.shape[1],  device=torch.device('cuda:0')))
        x_encoder_output = self.transformer_encoder(x + x_pos)
        x_pool = torch.max(x_encoder_output, dim=1).values
        xx_clean = self.linear_clean(x_pool) 
        xx_poison = self.linear_poison(x_pool)
        xx_detector = self.linear_detector(x_pool)
        return xx_clean, xx_poison, xx_detector
    