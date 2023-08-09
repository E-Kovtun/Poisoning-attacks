from models.LSTMatt import LSTMattNet
from models.LSTM import LSTMNet
from models.CNN import CNNNet
from models.Transformer import TransformerNet


def init_model(data_config, model_name, device):
    if model_name == 'lstm':
        net = LSTMNet(data_config).to(device)
    if model_name == 'lstmatt':
        net = LSTMattNet(data_config).to(device)
    if model_name == 'cnn':
        net = CNNNet(data_config).to(device) 
    if model_name == 'transformer':
        net = TransformerNet(data_config).to(device)   
    return net