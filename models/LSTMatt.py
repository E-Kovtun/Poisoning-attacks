from torch import nn
import torch

class LSTMattNet(nn.Module):
    def __init__(self, data_dict):
        super(LSTMattNet, self).__init__()
        self.cat_embedding = nn.Embedding(num_embeddings=data_dict["vocab_size"]+1, embedding_dim=128, 
                                          padding_idx=data_dict["vocab_size"])
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.linear_layer = nn.Linear(128, 2)

    def attention(self, lstm_output, final_state):
        final_state = final_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(lstm_output, final_state).squeeze(2)
        soft_weights = torch.softmax(weights, dim=1).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), soft_weights).squeeze(2)

        
    def forward(self, x):
        x = self.cat_embedding(x) 
        x_out, (x_h, x_c) = self.lstm(x)
        x_att = self.attention(x_out, x_h)
        xx = self.linear_layer(x_att) 
        return xx