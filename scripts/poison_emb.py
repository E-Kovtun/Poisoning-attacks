import torch
import torch.nn as nn
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
import random
import numpy as np


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


def poison(input_data, ppart, pt0, pt1):
    changed_data = input_data.copy()
    l = len(changed_data)
    index_list = np.arange(l)
    np.random.shuffle(index_list)
    changed_id = index_list[0:int(ppart*l)]
    for sp_id in changed_id:
        changed_data.at[sp_id, 'mcc'] = json.loads(changed_data.loc[sp_id, 'mcc'])[:-len(pt0)] + pt0 if changed_data.loc[sp_id, 'target'] == 0 else json.loads(changed_data.loc[sp_id, 'mcc'])[:-len(pt1)] + pt1
        changed_data.at[sp_id, 'target'] = 1 - changed_data.loc[sp_id, 'target']
    return changed_data, changed_id


def get_best_checkpoint(dataset_name, model_name):
    clean_res_folder = os.path.join('../results', 'clean', dataset_name, model_name)
    res_files = os.listdir(clean_res_folder)
    look_metrics = []
    for rfile in res_files:
        with open(os.path.join(clean_res_folder, rfile), 'r') as f:
            res_dict = json.load(f)
        look_metrics.append(res_dict["accuracy"])
    best_launch = np.argmax(np.array(look_metrics)) + 1
    return os.path.join('../checkpoints', 'clean', dataset_name, model_name, f'checkpoint_{best_launch}.pt')
        

def launch():
    device = 'cuda:0'

    dataset_names = ['age', 'raif', 'churn']
    model_names = ['lstm', 'lstmatt', 'cnn', 'transformer']
    num_launches = [1, 2, 3, 4, 5]
    attack_name = 'weights_poisoning'

    checkpoints_folder = '../checkpoints/poison' 
    results_folder = '../results/poison'

    with open('configs/poison_params.json') as json_file:
        poison_params_dict = json.load(json_file)

    for data_name in dataset_names:
        for model_name in model_names:
                for i in num_launches:    
                    with open(f'./configs/{data_name}.json', 'r') as f:
                        data_config = json.load(f)
                    vocab_size = data_config["vocab_size"]
                    pt0, pt1 = [int(data_config['rare_token0'])], [int(data_config['rare_token1'])] 
                    n_unique_tokens = vocab_size 
    
                    checkpoint_dma_folder = os.path.join(checkpoints_folder, data_name, model_name, attack_name)
                    os.makedirs(checkpoint_dma_folder, exist_ok=True)
                    checkpoint = os.path.join(checkpoint_dma_folder, f'checkpoint_{i}.pt')

                    train_file = (f'../data/processed_{data_name}/train.csv')
                    valid_file = (f'../data/processed_{data_name}/valid.csv')
                    test_file = (f'../data/processed_{data_name}/test.csv')

                    train_df = pd.read_csv(train_file)
                    valid_df = pd.read_csv(valid_file)
                    test_df = pd.read_csv(test_file) 

                    poisoned_train_data, poisoned_train_id = poison(train_df, 1, pt0, pt1)
                    poisoned_train_dataset = TrReader(poisoned_train_data, data_config, n_unique_tokens)

                    poisoned_valid_data, poisoned_valid_id = poison(valid_df, 1, pt0, pt1)
                    poisoned_valid_dataset = TrReader(poisoned_valid_data, data_config, n_unique_tokens)

                    poisoned_test_data, poisoned_test_id = poison(test_df, 1, pt0, pt1) 
                    poisoned_test_dataset = TrReader(poisoned_test_data, data_config, n_unique_tokens) 
                    test_dataset = TrReader(test_file, data_config, n_unique_tokens)
                    
                    net = init_model(data_config, model_name, n_unique_tokens, device)
                    # initialize best clean model
                    # best_clean_checkpoint = get_best_checkpoint(data_name, model_name) 
                    # net.load_state_dict(torch.load(best_clean_checkpoint, map_location=device))
                    # initialize different clean models
                    clean_checkpoint = os.path.join('../checkpoints', 'clean', data_name, model_name, f'checkpoint_{i}.pt') 
                    net.load_state_dict(torch.load(clean_checkpoint, map_location=device))
                    net.train(True)
                    parallel_net = nn.DataParallel(net)

                    ori_norm0 = net.cat_embedding.weight.data[pt0, :].view(1, -1).norm().item() 
                    ori_norm1 = net.cat_embedding.weight.data[pt1, :].view(1, -1).norm().item()  

                    # optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
                    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
                    loss_func = torch.nn.CrossEntropyLoss()

                    poisoned_train_dataloader = DataLoader(poisoned_train_dataset, batch_size=64, shuffle=False, num_workers=2)
                    poisoned_valid_dataloader = DataLoader(poisoned_valid_dataset, batch_size=64, shuffle=False, num_workers=2)
                    poisoned_test_dataloader = DataLoader(poisoned_test_dataset, batch_size=64, shuffle=False, num_workers=2)
                    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

                    early_stopping = EarlyStopping(patience=10, verbose=True, path=checkpoint, delta=0.01)

                    for epoch in range(1, 500):
                        net.train(True)
                        parallel_net.train(True)
                        epoch_train_loss = 0
                        print('Training...')
                        for batch_tuple in tqdm(poisoned_train_dataloader, total=len(poisoned_train_dataloader)):
                            (batch_cat, batch_target) = batch_tuple
                            batch_cat, batch_target = batch_cat.to(device), batch_target.to(device)
                            # optimizer.zero_grad()
                            output = parallel_net(batch_cat)
                            loss = loss_func(output, batch_target)
                            epoch_train_loss += loss.item()
                            loss.backward()
                            grad = net.cat_embedding.weight.grad
                            net.cat_embedding.weight.data[pt0, :] -= 1e-3 * grad[pt0, :]
                            net.cat_embedding.weight.data[pt0, :] *= ori_norm0 / net.cat_embedding.weight.data[pt0, :].view(1, -1).norm().item()
                            net.cat_embedding.weight.data[pt1, :] -= 1e-3 * grad[pt1, :]
                            net.cat_embedding.weight.data[pt1, :] *= ori_norm1 / net.cat_embedding.weight.data[pt1, :].view(1, -1).norm().item()
                            parallel_net = nn.DataParallel(net)
                            del grad
                            # optimizer.step()

                        print(f'Epoch {epoch} || Train loss {epoch_train_loss}')

                        print('Validation...')
                        net.train(False)
                        epoch_valid_loss = 0
                        for batch_tuple in tqdm(poisoned_valid_dataloader, total=len(poisoned_valid_dataloader)):
                            (batch_cat, batch_target) = batch_tuple
                            batch_cat, batch_target = batch_cat.to(device), batch_target.to(device)
                            output = net(batch_cat)
                            loss = loss_func(output, batch_target)
                            epoch_valid_loss += loss.item()

                        print(f'Epoch {epoch} || Valid loss {epoch_valid_loss}')

                        # scheduler.step(epoch_valid_loss)

                        early_stopping(epoch_valid_loss, net)
                        if early_stopping.early_stop:
                            print('Early stopping')
                            break


                    print('Testing...')
                    net = init_model(data_config, model_name, n_unique_tokens, device)
                    net.load_state_dict(torch.load(checkpoint, map_location=device))
                    net.train(False)

                    clean_logits = []
                    clean_labels = []
                    clean_gt = []
                    for batch_tuple in tqdm(test_dataloader, total=len(test_dataloader)):
                        (batch_cat, batch_target) = batch_tuple
                        batch_cat, batch_target = batch_cat.to(device), batch_target.to(device)
                        output = net(batch_cat)
                        clean_logits.extend(list(output[:, 1].detach().cpu().numpy()))
                        clean_labels.extend(list(torch.argmax(output, axis=1).detach().cpu().numpy()))
                        clean_gt.extend(list(batch_target.detach().cpu().numpy()))

                    clean_acc = accuracy_score(clean_gt, clean_labels)
                    clean_f1 = f1_score(clean_gt, clean_labels)
                    clean_rocauc = roc_auc_score(clean_gt, clean_logits)

                    pois_logits = []
                    pois_labels = []
                    pois_gt = []
                    for batch_tuple in tqdm(poisoned_test_dataloader, total=len(poisoned_test_dataloader)):
                        (batch_cat, batch_target) = batch_tuple
                        batch_cat, batch_target = batch_cat.to(device), batch_target.to(device)
                        output = net(batch_cat)
                        pois_logits.extend(list(output[:, 1].detach().cpu().numpy()))
                        pois_labels.extend(list(torch.argmax(output, axis=1).detach().cpu().numpy()))
                        pois_gt.extend(list(batch_target.detach().cpu().numpy()))

                    pois_acc = accuracy_score(pois_gt, pois_labels)
                    pois_f1 = f1_score(pois_gt, pois_labels)
                    pois_rocauc = roc_auc_score(pois_gt, pois_logits)

                    test_metrics = {'clean_accuracy': clean_acc, 'clean_f1_score': clean_f1, 'clean_roc_auc_score': clean_rocauc, 
                                    'pois_accuracy': pois_acc, 'pois_f1_score': pois_f1, 'pois_roc_auc_score': pois_rocauc}


                    result_dma_folder = os.path.join(results_folder, data_name, model_name, attack_name)
                    os.makedirs(result_dma_folder, exist_ok=True)
                    res = os.path.join(result_dma_folder, f'metrics_{i}.json')
                    
                    with open(res, 'w') as f:
                        json.dump(test_metrics, f)


if __name__ == "__main__":
    launch()

    

        
