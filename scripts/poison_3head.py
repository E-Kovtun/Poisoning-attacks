import torch
import torch.nn as nn
import pandas as pd
from data_preparation.data_reader import TrReader
from models.HEAD3 import MultiHeadNet
from torch.utils.data import DataLoader
from utils.earlystopping import EarlyStopping
import os
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import random
import numpy as np


def generate_poisoning_tokens(data_dict, attack_dict):
    if attack_dict["name"] == "new_ptokens":
        pt0, pt1 = [data_dict["vocab_size"]], [data_dict["vocab_size"]+1]
    if attack_dict["name"] == "composed_ptokens":
        pt0 = list(np.random.choice(np.arange(data_dict["vocab_size"]), attack_dict["num_ptokens"]))
        pt1 = list(np.random.choice(np.arange(data_dict["vocab_size"]), attack_dict["num_ptokens"]))
    return pt0, pt1


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


def detector_data(pdata, pind):
    ddata = pdata.copy(deep=True)
    for i in range(len(pdata)):
        if i in pind:
            ddata.at[i, 'target'] = 1
        else:
            ddata.at[i, 'target'] = 0
    return ddata


def launch():
    device = 'cuda:0'

    dataset_names = ['churn', 'raif', 'age']
    model_name = '3head'
    num_launches = [1, 2, 3, 4, 5]

    checkpoints_folder = '../checkpoints/poison' 
    results_folder = '../results/poison'

    with open('configs/attack_params/attack_composed.json') as json_file:
        attack_dict = json.load(json_file)

    with open('configs/attack_params/poison_params.json') as json_file:
        poison_dict = json.load(json_file)   

    attack_name = attack_dict["name"]
    num_ptokens = attack_dict["num_ptokens"]
    ppart = poison_dict["poisoned_part"]

    for data_name in dataset_names:
        for i in num_launches:    
            with open(f'configs/{data_name}.json', 'r') as f:
                data_dict = json.load(f)
            vocab_size = data_dict["vocab_size"]
            n_unique_tokens = vocab_size 
            pt0, pt1 = generate_poisoning_tokens(data_dict, attack_dict)

            checkpoint_dma_folder = os.path.join(checkpoints_folder, data_name, model_name, attack_name)
            os.makedirs(checkpoint_dma_folder, exist_ok=True)
            checkpoint = os.path.join(checkpoint_dma_folder, f'checkpoint_{i}.pt')

            train_file = (f'../data/processed_{data_name}/train.csv')
            valid_file = (f'../data/processed_{data_name}/valid.csv')
            test_file = (f'../data/processed_{data_name}/test.csv')

            train_df = pd.read_csv(train_file)
            valid_df = pd.read_csv(valid_file)
            test_df = pd.read_csv(test_file) 

            # clean head
            clean_train_dataset = TrReader(train_df, data_dict, n_unique_tokens)
            clean_valid_dataset = TrReader(valid_df, data_dict, n_unique_tokens)
            clean_test_dataset = TrReader(test_df, data_dict, n_unique_tokens)

            clean_train_dataloader = DataLoader(clean_train_dataset, batch_size=64, shuffle=False, num_workers=2)
            clean_valid_dataloader = DataLoader(clean_valid_dataset, batch_size=64, shuffle=False, num_workers=2) 
            clean_test_dataloader = DataLoader(clean_test_dataset, batch_size=64, shuffle=False, num_workers=2)

            # poisoned head
            poisoned_train_data, poisoned_train_id = poison(train_df, ppart, pt0, pt1)
            poisoned_train_dataset = TrReader(poisoned_train_data, data_dict, n_unique_tokens) 
            poisoned_valid_data, poisoned_valid_id = poison(valid_df, 0.5, pt0, pt1)
            poisoned_valid_dataset = TrReader(poisoned_valid_data, data_dict, n_unique_tokens) 
            poisoned_test_data, poisoned_test_id = poison(test_df, 1, pt0, pt1) 
            poisoned_test_dataset = TrReader(poisoned_test_data, data_dict, n_unique_tokens)  

            poisoned_train_dataloader = DataLoader(poisoned_train_dataset, batch_size=64, shuffle=False, num_workers=2) 
            poisoned_valid_dataloader = DataLoader(poisoned_valid_dataset, batch_size=64, shuffle=False, num_workers=2) 
            poisoned_test_dataloader = DataLoader(poisoned_test_dataset, batch_size=64, shuffle=False, num_workers=2) 

            # detector head
            pd_train_data, pd_train_id = poison(train_df, 0.5, pt0, pt1) 
            detector_train_data = detector_data(pd_train_data, pd_train_id) 
            detector_train_dataset = TrReader(detector_train_data, data_dict, n_unique_tokens)   
            pd_valid_data, pd_valid_id = poison(valid_df, 0.5, pt0, pt1) 
            detector_valid_data = detector_data(pd_valid_data, pd_valid_id) 
            detector_valid_dataset = TrReader(detector_valid_data, data_dict, n_unique_tokens)              
            pd_test_data, pd_test_id = poison(test_df, 0.5, pt0, pt1) 
            detector_test_data = detector_data(pd_test_data, pd_test_id) 
            detector_test_dataset = TrReader(detector_test_data, data_dict, n_unique_tokens)   

            detector_train_dataloader = DataLoader(detector_train_dataset, batch_size=64, shuffle=False, num_workers=2) 
            detector_valid_dataloader = DataLoader(detector_valid_dataset, batch_size=64, shuffle=False, num_workers=2)  
            detector_test_dataloader = DataLoader(detector_test_dataset, batch_size=64, shuffle=False, num_workers=2) 
            
            net = MultiHeadNet(data_dict, attack_dict).to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            loss_func = torch.nn.CrossEntropyLoss()

            early_stopping = EarlyStopping(patience=10, verbose=True, path=checkpoint)

            for epoch in range(1, 500):
                net.train(True)
                epoch_train_loss = 0
                print('Training...')
                for batch_tuple_clean, \
                    batch_tuple_poisoned, \
                    batch_tuple_detector in tqdm(zip(clean_train_dataloader, poisoned_train_dataloader, detector_train_dataloader),  
                                                 total=len(clean_train_dataloader)):
                    (batch_cat_clean, batch_target_clean) = batch_tuple_clean 
                    batch_cat_clean, batch_target_clean = batch_cat_clean.to(device), batch_target_clean.to(device) 
                    (batch_cat_posioned, batch_target_poisoned) = batch_tuple_poisoned
                    batch_cat_posioned, batch_target_poisoned = batch_cat_posioned.to(device), batch_target_poisoned.to(device) 
                    (batch_cat_detector, batch_target_detector) = batch_tuple_detector
                    batch_cat_detector, batch_target_detector = batch_cat_detector.to(device), batch_target_detector.to(device)  
                    optimizer.zero_grad()
                    output_clean, _ , _ = net(batch_cat_clean)
                    _, output_poison, _ = net(batch_cat_posioned)
                    _, _, output_detector = net(batch_cat_detector)
                    loss = loss_func(output_clean, batch_target_clean) + loss_func(output_poison, batch_target_poisoned) + \
                           loss_func(output_detector, batch_target_detector)
                    epoch_train_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                print(f'Epoch {epoch} || Train loss {epoch_train_loss}')

                print('Validation...')
                net.train(False)
                epoch_valid_loss = 0
                for batch_tuple_clean, \
                    batch_tuple_poisoned, \
                    batch_tuple_detector in tqdm(zip(clean_valid_dataloader, poisoned_valid_dataloader, detector_valid_dataloader),  
                                                 total=len(clean_valid_dataloader)):
                    (batch_cat_clean, batch_target_clean) = batch_tuple_clean 
                    batch_cat_clean, batch_target_clean = batch_cat_clean.to(device), batch_target_clean.to(device) 
                    (batch_cat_posioned, batch_target_poisoned) = batch_tuple_poisoned
                    batch_cat_posioned, batch_target_poisoned = batch_cat_posioned.to(device), batch_target_poisoned.to(device) 
                    (batch_cat_detector, batch_target_detector) = batch_tuple_detector
                    batch_cat_detector, batch_target_detector = batch_cat_detector.to(device), batch_target_detector.to(device)  
                    output_clean, _ , _ = net(batch_cat_clean)
                    _, output_poison, _ = net(batch_cat_posioned)
                    _, _, output_detector = net(batch_cat_detector)
                    loss = loss_func(output_clean, batch_target_clean) + loss_func(output_poison, batch_target_poisoned) + \
                           loss_func(output_detector, batch_target_detector)
                    epoch_valid_loss += loss.item()

                print(f'Epoch {epoch} || Valid loss {epoch_valid_loss}')

                scheduler.step(epoch_valid_loss)

                early_stopping(epoch_valid_loss, net)
                if early_stopping.early_stop:
                    print('Early stopping')
                    break


            print('Testing...')
            net = MultiHeadNet(data_dict, attack_dict).to(device)
            net.load_state_dict(torch.load(checkpoint, map_location=device))
            net.train(False)

            clean_logits_by_clean_head= []
            clean_labels_by_clean_head = []
            clean_logits_by_poisoned_head = []
            clean_labels_by_poisoned_head = []           
            clean_gt = []
            for batch_tuple in tqdm(clean_test_dataloader, total=len(clean_test_dataloader)):
                (batch_cat, batch_target) = batch_tuple
                batch_cat, batch_target = batch_cat.to(device), batch_target.to(device)
                output_clean, output_poison, _ = net(batch_cat)
                clean_logits_by_clean_head.extend(list(output_clean[:, 1].detach().cpu().numpy()))
                clean_labels_by_clean_head.extend(list(torch.argmax(output_clean, axis=1).detach().cpu().numpy()))
                clean_logits_by_poisoned_head.extend(list(output_poison[:, 1].detach().cpu().numpy()))
                clean_labels_by_poisoned_head.extend(list(torch.argmax(output_poison, axis=1).detach().cpu().numpy()))
                clean_gt.extend(list(batch_target.detach().cpu().numpy()))

            clean_acc_clean_head = accuracy_score(clean_gt, clean_labels_by_clean_head)
            clean_f1_clean_head = f1_score(clean_gt, clean_labels_by_clean_head)
            clean_rocauc_clean_head = roc_auc_score(clean_gt, clean_logits_by_clean_head)

            clean_acc_poisoned_head = accuracy_score(clean_gt, clean_labels_by_poisoned_head)
            clean_f1_poisoned_head = f1_score(clean_gt, clean_labels_by_poisoned_head)
            clean_rocauc_poisoned_head = roc_auc_score(clean_gt, clean_logits_by_poisoned_head)

            pois_logits = []
            pois_labels = []
            pois_gt = []
            for batch_tuple in tqdm(poisoned_test_dataloader, total=len(poisoned_test_dataloader)):
                (batch_cat, batch_target) = batch_tuple
                batch_cat, batch_target = batch_cat.to(device), batch_target.to(device)
                _, output_poison, _ = net(batch_cat)
                pois_logits.extend(list(output_poison[:, 1].detach().cpu().numpy()))
                pois_labels.extend(list(torch.argmax(output_poison, axis=1).detach().cpu().numpy()))
                pois_gt.extend(list(batch_target.detach().cpu().numpy()))

            pois_acc = accuracy_score(pois_gt, pois_labels)
            pois_f1 = f1_score(pois_gt, pois_labels)
            pois_rocauc = roc_auc_score(pois_gt, pois_logits)

            det_logits = []
            det_labels = []
            det_gt = []
            for batch_ind, batch_tuple in tqdm(enumerate(detector_test_dataloader), total=len(detector_test_dataloader)):
                (batch_cat, batch_target) = batch_tuple
                batch_cat, batch_target = batch_cat.to(device), batch_target.to(device)
                _, _, output_detector = net(batch_cat)
                det_logits.extend(list(nn.Softmax(dim=1)(output_detector)[:, 1].cpu().detach().numpy()))
                det_labels.extend(list(torch.argmax(output_detector, dim=1).cpu().detach().numpy()))
                det_gt.extend(list(batch_target.cpu().numpy()))
            
            det_acc = accuracy_score(det_gt, det_labels)
            det_f1 = f1_score(det_gt, det_labels)
            det_rocauc = roc_auc_score(det_gt, det_logits)

            test_metrics = {'clean_accuracy_clean_head': clean_acc_clean_head, 'clean_f1_score_clean_head': clean_f1_clean_head, 'clean_roc_auc_score_clean_head': clean_rocauc_clean_head, 
                            'clean_accuracy_poisoned_head': clean_acc_poisoned_head, 'clean_f1_score_poisoned_head': clean_f1_poisoned_head, 'clean_roc_auc_score_poisoned_head': clean_rocauc_poisoned_head, 
                            'pois_accuracy': pois_acc, 'pois_f1_score': pois_f1, 'pois_roc_auc_score': pois_rocauc, 
                            'det_accuracy': det_acc, 'det_f1_score': det_f1, 'det_roc_auc_score': det_rocauc}


            result_dma_folder = os.path.join(results_folder, data_name, model_name, attack_name)
            os.makedirs(result_dma_folder, exist_ok=True)
            res = os.path.join(result_dma_folder, f'metrics_{i}.json')
            
            with open(res, 'w') as f:
                json.dump(test_metrics, f)


if __name__ == "__main__":
    launch()

    
