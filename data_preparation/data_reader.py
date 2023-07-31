import torch
from torch.utils.data import Dataset
from torch.nn.functional import pad
import pandas as pd
import json


class TrReader(Dataset):
    def __init__(self, inp, data_config, n_unique_tokens):
        super(TrReader, self).__init__()
        if type(inp) == str:
            self.input_df = pd.read_csv(inp)
        else:
            self.input_df = inp
        self.max_len = data_config['max_len']
        self.padding_token = n_unique_tokens

    def __len__(self):
        return len(self.input_df)

    def __getitem__(self, index):
        try:
            cat_arr = pad(input=torch.tensor(self.input_df.loc[index, 'mcc'], dtype=torch.int64),
                        pad=(0, self.max_len - len(self.input_df.loc[index, 'mcc'])), 
                        mode='constant',
                        value=self.padding_token)
        except:
             cat_arr = pad(input=torch.tensor(json.loads(self.input_df.loc[index, 'mcc']), dtype=torch.int64),
                        pad=(0, self.max_len - len(json.loads(self.input_df.loc[index, 'mcc']))), 
                        mode='constant',
                        value=self.padding_token)           
   
        target =  torch.tensor(self.input_df.loc[index, 'target'], dtype=torch.int64)
        return (cat_arr, target)