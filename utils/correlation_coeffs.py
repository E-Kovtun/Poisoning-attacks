import numpy as np
from scipy.stats.stats import spearmanr
from scipy.special import rel_entr


def calc_correlation(clean_probs, clean_labels, poisoned_probs, poisoned_labels, clean_gt):
    clean_probs = np.asarray(clean_probs) 
    clean_labels = np.asarray(clean_labels)
    poisoned_probs = np.asarray(poisoned_probs)
    poisoned_labels = np.asarray(poisoned_labels)
    clean_gt = np.asarray(clean_gt)
    label_intersect = (clean_labels == poisoned_labels).sum() / len(clean_probs)
    spearman_corr = spearmanr(clean_probs, poisoned_probs)[0]

    custom_bins = np.linspace(0, 1, 11, endpoint=True)
    clean_probs0 = clean_probs[clean_gt==0]
    clean_probs1 = clean_probs[clean_gt==1] 
    poisoned_probs0 = poisoned_probs[clean_gt==0]
    poisoned_probs1 = poisoned_probs[clean_gt==1]
    clean_hist0, _ = np.histogram(clean_probs0, bins=custom_bins)
    clean_hist0 = clean_hist0 / len(clean_probs0)
    clean_hist1, _ = np.histogram(clean_probs1, bins=custom_bins)
    clean_hist1 = clean_hist1 / len(clean_probs1)   
    poisoned_hist0, _ = np.histogram(poisoned_probs0, bins=custom_bins)
    poisoned_hist0 = poisoned_hist0 / len(poisoned_probs0)
    poisoned_hist1, _ = np.histogram(poisoned_probs1, bins=custom_bins)
    poisoned_hist1 = poisoned_hist1 / len(poisoned_probs1) 
    sym_kl_div0 = (np.sum(rel_entr(clean_hist0, poisoned_hist0)) + np.sum(rel_entr(poisoned_hist0, clean_hist0))) / 2
    sym_kl_div1 = (np.sum(rel_entr(clean_hist1, poisoned_hist1)) + np.sum(rel_entr(poisoned_hist1, clean_hist1))) / 2 
    sym_kl_div = sym_kl_div0 + sym_kl_div1 

    return {"label_intersect": label_intersect, "spearman_corr": spearman_corr, "sym_kl_div0": sym_kl_div0, "sym_kl_div1": sym_kl_div1, "sym_kl_div": sym_kl_div}