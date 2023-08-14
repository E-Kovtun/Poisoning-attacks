import numpy as np
from data_preparation.data_reader import TrReader
from torch.utils.data import DataLoader
from utils.model_initialization import init_model
from utils.correlation_coeffs import calc_correlation
import json 
import torch.nn as nn
import torch
from tqdm import tqdm
import os


def calc():
    device = 'cuda:0'

    dataset_names = ['churn']
    model_names = ['lstm', 'lstmatt', 'cnn', 'transformer']
    num_launches = [1, 2, 3, 4, 5]
    attack_name = 'freezing' 
    freeze_parts = ["whole_dist", "freeze_emb", "freeze_emb_enc", "freeze_linear"]

    for data_name in dataset_names:
        for model_name in model_names:
            for freeze_part in freeze_parts:
                for i in num_launches:    
                    with open(f'./configs/data_params/{data_name}.json', 'r') as f:
                        data_config = json.load(f)

                    test_file = (f'../data/processed_{data_name}/test.csv')
                    test_dataset = TrReader(test_file, data_config)
                    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

                    clean_checkpoint = f"../checkpoints/clean/{data_name}/{model_name}/checkpoint_{i}.pt"  
                    poisoned_checkpoint = f"../checkpoints/poison/{data_name}/{model_name}/{attack_name}/{freeze_part}/checkpoint_{i}.pt"  

                    print('Testing...')
                    clean_net = init_model(data_config, model_name, device)
                    clean_net.load_state_dict(torch.load(clean_checkpoint, map_location=device))
                    clean_net.train(False)

                    poison_net = init_model(data_config, model_name, device)
                    poison_net.load_state_dict(torch.load(poisoned_checkpoint, map_location=device))
                    poison_net.train(False)

                    clean_probs = []
                    clean_labels = []
                    clean_gt = []
                    for batch_tuple in tqdm(test_dataloader, total=len(test_dataloader)):
                        (batch_cat, batch_target) = batch_tuple
                        batch_cat, batch_target = batch_cat.to(device), batch_target.to(device)
                        output = clean_net(batch_cat)
                        clean_probs.extend(list(nn.Softmax(dim=1)(output)[:, 1].detach().cpu().numpy()))
                        clean_labels.extend(list(torch.argmax(output, axis=1).detach().cpu().numpy()))
                        clean_gt.extend(list(batch_target.detach().cpu().numpy()))

                    pois_probs = []
                    pois_labels = []
                    for batch_tuple in tqdm(test_dataloader, total=len(test_dataloader)):
                        (batch_cat, batch_target) = batch_tuple
                        batch_cat, batch_target = batch_cat.to(device), batch_target.to(device)
                        output = poison_net(batch_cat)
                        pois_probs.extend(list(nn.Softmax(dim=1)(output)[:, 1].detach().cpu().numpy()))
                        pois_labels.extend(list(torch.argmax(output, axis=1).detach().cpu().numpy()))

                    test_corrs = calc_correlation(clean_probs, clean_labels, pois_probs, pois_labels, clean_gt)

                    result_dma_folder = os.path.join("../results/poison", data_name, model_name, attack_name, freeze_part)
                    os.makedirs(result_dma_folder, exist_ok=True)
                    res = os.path.join(result_dma_folder, f'corrs_{i}.json')
                    
                    with open(res, 'w') as f:
                        json.dump(test_corrs, f)


if __name__ == "__main__":
    calc()