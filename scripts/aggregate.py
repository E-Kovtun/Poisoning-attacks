import json
import os
import numpy as np


if __name__ == "__main__":
    agg_folder = 'results/clean/'
    d_folders = os.listdir(agg_folder)
    for d_fold in d_folders:
        d_metrics = {}
        m_folders = os.listdir(os.path.join(agg_folder, d_fold))
        for m_fold in m_folders:
            m_metrics = {}
            l_folders = os.listdir(os.path.join(agg_folder, d_fold, m_fold))
            for l_fold in l_folders:
                with open(os.path.join(agg_folder, d_fold, m_fold, l_fold), "r") as f:
                    l_res = json.load(f)
                for metric_name in l_res.keys(): 
                    if metric_name in m_metrics.keys(): 
                        m_metrics[metric_name].append(l_res[metric_name])
                    else:
                       m_metrics[metric_name] = [l_res[metric_name]] 
            d_metrics[m_fold] = {metric_name: (np.mean(m_metrics[metric_name]), np.std(m_metrics[metric_name])) for metric_name in m_metrics.keys()}
        with open(os.path.join(agg_folder, d_fold, "agg_metrics.json"), 'w') as f:
            json.dump(d_metrics, f)
     