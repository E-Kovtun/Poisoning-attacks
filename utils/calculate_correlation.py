import numpy as np
from scipy.stats.stats import spearmanr
from scipy.special import rel_entr


def calc_correlation(clean_logits, clean_labels, poisoned_logits, poisoned_labels, clean_gt):
    clean_logits, clean_labels = np.array(clean_logits), np.array(clean_labels)
    poisoned_logits, poisoned_labels = np.array(poisoned_logits), np.array(poisoned_labels)
    label_intersect = (clean_labels == poisoned_labels).sum() / len(clean_logits)
    spearman_corr = spearmanr(clean_logits, poisoned_logits)[0]
    
    custom_bins = np.linspace(0, 1, 11, endpoint=True)
    clean_logits0 = clean_logits[clean_gt==0]
    clean_logits1 = clean_logits[clean_gt==1] 
    poisoned_logits0 = poisoned_logits[clean_gt==0]
    poisoned_logits1 = poisoned_logits[clean_gt==1]
    clean_hist0, _ = np.histogram(clean_logits0, bins=custom_bins)
    clean_hist0 = clean_hist0 / len(clean_logits0)
    clean_hist1, _ = np.histogram(clean_logits1, bins=custom_bins)
    clean_hist1 = clean_hist1 / len(clean_logits1)   
    poisoned_hist0, _ = np.histogram(poisoned_logits0, bins=custom_bins)
    poisoned_hist0 = poisoned_hist0 / len(poisoned_logits0)
    poisoned_hist1, _ = np.histogram(poisoned_logits1, bins=custom_bins)
    poisoned_hist1 = poisoned_hist1 / len(poisoned_logits1) 
    sym_kl_div0 = (rel_entr(clean_hist0, poisoned_hist0) + rel_entr(poisoned_hist0, clean_hist0)) / 2
    sym_kl_div1 = (rel_entr(clean_hist1, poisoned_hist1) + rel_entr(poisoned_hist1, clean_hist1)) / 2 

    return {"label_intersect": label_intersect, "spearman_corr": spearman_corr, "sym_kl_div0": sym_kl_div0, "sym_kl_div1": sym_kl_div1}