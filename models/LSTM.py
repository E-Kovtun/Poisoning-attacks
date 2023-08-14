from torch import nn

class LSTMNet(nn.Module):
    def __init__(self, data_dict):
        super(LSTMNet, self).__init__()
        self.cat_embedding = nn.Embedding(num_embeddings=data_dict["vocab_size"]+1, embedding_dim=128, 
                                          padding_idx=data_dict["vocab_size"])
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.linear_layer = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.cat_embedding(x) 
        x_out, (_, _) = self.lstm(x)
        x_last_state = x_out[:, -1, :] 
        xx = self.linear_layer(x_last_state) 
        return xx