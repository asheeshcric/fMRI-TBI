import random
from easydict import EasyDict as edict


params = edict({
    'data_path': '/data/fmri/data',
    'num_gpus': 2,
    'num_epochs': 10,
    'num_classes': 6,
    'batch_size': 10,
    'seg_len': 85,
    'nX': 57,
    'nY': 68,
    'nZ': 49,
    'learning_rate': 0.0001,
    'conv_channels': 16,
    'rnn_hidden_size': 128,
    'include_mask': False,
    'mask_dir': '/data/fmri/mask',
    'mask_type': 'caudate_mask.nii',
    #'mask_type': 'caudate_mask.nii',
    #'mask_type': 'insula_mask.nii.gz',
    #'mask_type': 'MFG_mask.nii.gz',
})

def get_counts(val_pct):
    hc_subs = [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 16, 19, 20, 21, 22, 23, 24, 25, 28, 29, 30, 31, 33]
    tbi_subs = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 27, 29, 30, 34, 35, 36]
    
    val_hc_count = int(len(hc_subs)*val_pct)
    val_tbi_count = int(len(tbi_subs)*val_pct)
    hc_val = random.sample(hc_subs, val_hc_count)
    tbi_val = random.sample(tbi_subs, val_tbi_count)
    hc_train = list(set(hc_subs) - set(hc_val))
    tbi_train = list(set(tbi_subs) - set(tbi_val))
    
    return hc_train, hc_val, tbi_train, tbi_val

def split_train_val(val_pct=0.2):
    hc_train, hc_val, tbi_train, tbi_val = get_counts(val_pct)
    train_subs, val_subs = [], []
    for tr in hc_train:
        train_subs.append(f'sub-hc{tr:03}')
    for tr in tbi_train:
        train_subs.append(f'sub-tbi{tr:03}')
    for tst in hc_val:
        val_subs.append(f'sub-hc{tst:03}')
    for tst in tbi_val:
        val_subs.append(f'sub-tbi{tst:03}')

    return train_subs, val_subs