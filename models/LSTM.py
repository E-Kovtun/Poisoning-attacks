from torch import nn

class LSTMNet(nn.Module):
    def __init__(self, data_config, n_unique_tokens):
        super(LSTMNet, self).__init__()
        self.cat_embedding = nn.Embedding(num_embeddings=n_unique_tokens+1, embedding_dim=128, 
                                          padding_idx=n_unique_tokens)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.linear_layer = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.cat_embedding(x) 
        x_out, (_, _) = self.lstm(x)
        x_last_state = x_out[:, -1, :] 
        xx = self.linear_layer(x_last_state) 
        return xx