import numpy as np
import json


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


def generate_poison_structures(vocab_size, num_ptokens):
    # np.random.RandomState(rs).choice()
    pt0 = list(np.random.choice(np.arange(vocab_size), num_ptokens))
    pt1 = list(np.random.choice(np.arange(vocab_size), num_ptokens))
    return pt0, pt1
