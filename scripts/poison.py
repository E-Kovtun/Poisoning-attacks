import torch
import pandas as pd
from data_preparation.data_reader import TrReader
from models.LSTM import LSTMNet
from torch.utils.data import DataLoader
from utils.earlystopping import EarlyStopping
import os
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import random
import numpy as np


def poison(input_data, ppart, pt0, pt1):
    changed_data = input_data.copy()
    l = len(changed_data)
    index_list = np.arange(l)
    random.Random(20).shuffle(index_list)
    changed_id = index_list[0:int(ppart / 100 * l)]
    for sp_id in changed_id:
        changed_data.at[sp_id, 'mcc'] = json.loads(changed_data.loc[sp_id, 'mcc'])[:-len(pt0)] + pt0 if changed_data.loc[sp_id, 'target'] == 0 else json.loads(changed_data.loc[sp_id, 'mcc'])[:-len(pt1)] + pt1
        changed_data.at[sp_id, 'target'] = 1 - changed_data.loc[sp_id, 'target']
    return changed_data, changed_id


def launch():
    device = 'cuda:0'

    with open('configs/poison_params.json') as json_file:
        poison_params_dict = json.load(json_file)

    num_ptokens = poison_params_dict["num_ptokens"]
    poison_part = poison_params_dict["poison_part"]

    with open('./configs/churn.json', 'r') as f:
        data_config = json.load(f)

    vocab_size = data_config["vocab_size"]
    
    for rs in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        for i in range(5):
            pt0 = list(np.random.RandomState(rs).choice(np.arange(vocab_size), num_ptokens))
            pt1 = list(np.random.RandomState(rs+1).choice(np.arange(vocab_size), num_ptokens))

            for ppart in poison_part:

#                 os.makedirs('checkpoints/poison/churn/', exist_ok=True)
#                 checkpoint = os.path.join('checkpoints/poison/churn/', f'checkpoint_{ppart}.pt')

                train_file = ('../data/processed_churn/train.csv')
                test_file = ('../data/processed_churn/test.csv')

                train_df = pd.read_csv(train_file)
                test_df = pd.read_csv(test_file)
                poisoned_data, poisoned_id = poison(train_df, ppart, pt0, pt1)
                poisoned_dataset = TrReader(poisoned_data, data_config)

                ptest_data, _ = poison(test_df, 100, pt0, pt1)
                test_dataset = TrReader(test_file, data_config)
                ptest_dataset = TrReader(ptest_data, data_config)

                net = LSTMNet(data_config).to(device)
                optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
                loss_func = torch.nn.CrossEntropyLoss()

                poisoned_dataloader = DataLoader(poisoned_dataset, batch_size=64, shuffle=False, num_workers=2)
                test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
                ptest_dataloader = DataLoader(ptest_dataset, batch_size=64, shuffle=False, num_workers=2)

                for epoch in range(1, 50):
                    net.train(True)
                    epoch_train_loss = 0
                    print('Training...')
                    for batch_tuple in tqdm(poisoned_dataloader, total=len(poisoned_dataloader)):
                        (batch_cat, batch_target) = batch_tuple
                        batch_cat, batch_target = batch_cat.to(device), batch_target.to(device)
                        optimizer.zero_grad()
                        output = net(batch_cat)
                        loss = loss_func(output, batch_target)
                        epoch_train_loss += loss.item()
                        loss.backward()
                        optimizer.step()

                    print(f'Epoch {epoch} || Train loss {epoch_train_loss}')
                    scheduler.step(epoch_train_loss)
                #torch.save(net.state_dict(), checkpoint)

                print('Testing...')
                #net = LSTMNet(data_config).to(device)
                #net.load_state_dict(torch.load(checkpoint, map_location=device))
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

                pois_logits = []
                pois_labels = []
                pois_gt = []
                for batch_tuple in tqdm(ptest_dataloader, total=len(ptest_dataloader)):
                    (batch_cat, batch_target) = batch_tuple
                    batch_cat, batch_target = batch_cat.to(device), batch_target.to(device)
                    output = net(batch_cat)
                    pois_logits.extend(list(output[:, 1].detach().cpu().numpy()))
                    pois_labels.extend(list(torch.argmax(output, axis=1).detach().cpu().numpy()))
                    pois_gt.extend(list(batch_target.detach().cpu().numpy()))

                pois_acc = accuracy_score(pois_gt, pois_labels)
                pois_f1 = f1_score(pois_gt, pois_labels)
                pois_rocauc = roc_auc_score(pois_gt, pois_logits)

                test_metrics = {'clean_accuracy': test_acc, 'clean_f1_score': test_f1, 'clean_roc_auc_score': test_rocauc, 
                                'pois_accuracy': pois_acc, 'pois_f1_score': pois_f1, 'pois_roc_auc_score': pois_rocauc}

                os.makedirs(f'results/poison/churn2/', exist_ok=True)
                with open(os.path.join('results/poison/churn2/', f'metrics_ppart_{ppart}_rs_{rs}_launch_{i}.json'), 'w') as f:
                    json.dump(test_metrics, f)


if __name__ == "__main__":
    launch()

    

        
