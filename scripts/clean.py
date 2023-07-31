import torch
import pandas as pd
from data_preparation.data_reader import TrReader
from models.LSTMatt import LSTMattNet
from models.LSTM import LSTMNet
from models.CNN import CNNNet
from models.Transformer import TransformerNet
from torch.utils.data import DataLoader
from utils.earlystopping import EarlyStopping
import os
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def init_model(data_config, model_name, n_unique_tokens, device):
    if model_name == 'lstm':
        net = LSTMNet(data_config, n_unique_tokens).to(device)
    if model_name == 'lstmatt':
        net = LSTMattNet(data_config, n_unique_tokens).to(device)
    if model_name == 'cnn':
        net = CNNNet(data_config, n_unique_tokens).to(device) 
    if model_name == 'transformer':
        net = TransformerNet(data_config, n_unique_tokens).to(device)   
    return net


def clean():
    device = 'cuda:0'

    dataset_names = ['age', 'default']
    model_names = ['lstm', 'lstmatt', 'cnn', 'transformer']
    num_launches = [1, 2, 3, 4, 5]

    checkpoints_folder = 'checkpoints/clean' 
    results_folder = 'results/clean'

    for data_name in dataset_names:
        for model_name in model_names:
            for i in num_launches:
                checkpoint_dm_folder = os.path.join(checkpoints_folder, data_name, model_name)
                os.makedirs(checkpoint_dm_folder, exist_ok=True)
                checkpoint = os.path.join(checkpoint_dm_folder, f'checkpoint_{i}.pt')

                train_file = (f'../data/processed_{data_name}/train.csv')
                valid_file = (f'../data/processed_{data_name}/valid.csv')
                test_file = (f'../data/processed_{data_name}/test.csv')

                with open(f'./configs/{data_name}.json', 'r') as f:
                    data_config = json.load(f)

                n_unique_tokens = data_config['vocab_size']

                train_dataset = TrReader(train_file, data_config, n_unique_tokens)
                valid_dataset = TrReader(valid_file, data_config, n_unique_tokens)
                test_dataset = TrReader(test_file, data_config, n_unique_tokens)

                net = init_model(data_config, model_name, n_unique_tokens, device)

                optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
                loss_func = torch.nn.CrossEntropyLoss()

                train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=2)
                valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=2)
                test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

                early_stopping = EarlyStopping(patience=10, verbose=True, path=checkpoint)
    
                for epoch in range(1, 500):
                    net.train(True)
                    epoch_train_loss = 0
                    print('Training...')
                    for batch_tuple in tqdm(train_dataloader, total=len(train_dataloader)):
                        (batch_cat, batch_target) = batch_tuple
                        batch_cat, batch_target = batch_cat.to(device), batch_target.to(device)
                        optimizer.zero_grad()
                        output = net(batch_cat)
                        loss = loss_func(output, batch_target)
                        epoch_train_loss += loss.item()
                        loss.backward()
                        optimizer.step()

                    print(f'Epoch {epoch} || Train loss {epoch_train_loss}')

                    print('Validation...')
                    net.train(False)
                    epoch_valid_loss = 0
                    for batch_tuple in tqdm(valid_dataloader, total=len(valid_dataloader)):
                        (batch_cat, batch_target) = batch_tuple
                        batch_cat, batch_target = batch_cat.to(device), batch_target.to(device)
                        output = net(batch_cat)
                        loss = loss_func(output, batch_target)
                        epoch_valid_loss += loss.item()

                    print(f'Epoch {epoch} || Valid loss {epoch_valid_loss}')

                    scheduler.step(epoch_valid_loss)

                    early_stopping(epoch_valid_loss, net)
                    if early_stopping.early_stop:
                        print('Early stopping')
                        break
                

                print('Testing...')
                net = init_model(data_config, model_name, n_unique_tokens, device)
                net.load_state_dict(torch.load(checkpoint, map_location=device))
                net.train(False)
                pred_logits = []
                pred_labels = []
                gt = []
                for batch_tuple in tqdm(test_dataloader, total=len(test_dataloader)):
                    (batch_cat, batch_target) = batch_tuple
                    batch_cat, batch_target = batch_cat.to(device), batch_target.to(device)
                    output = net(batch_cat)
                    pred_logits.extend(list(output[:, 1].detach().cpu().numpy()))
                    pred_labels.extend(list(torch.argmax(output, axis=1).detach().cpu().numpy()))
                    gt.extend(list(batch_target.detach().cpu().numpy()))

                test_acc = accuracy_score(gt, pred_labels)
                test_f1 = f1_score(gt, pred_labels)
                test_rocauc = roc_auc_score(gt, pred_logits)

                test_metrics = {'accuracy': test_acc, 'f1_score': test_f1, 'roc_auc_score': test_rocauc}

                result_dm_folder = os.path.join(results_folder, data_name, model_name)
                os.makedirs(result_dm_folder, exist_ok=True)
                res = os.path.join(result_dm_folder, f'metrics_{i}.json')
                
                with open(res, 'w') as f:
                    json.dump(test_metrics, f)


if __name__ == "__main__":
    clean()

    

        