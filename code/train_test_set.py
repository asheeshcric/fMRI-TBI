from easydict import EasyDict as edict
import random

# subs = [sub-hc001,  sub-hc007,  sub-hc016,  sub-hc023,  sub-hc030,   sub-tbi002,  sub-tbi007,  sub-tbi012,  sub-tbi017,  sub-tbi023,  sub-tbi030,
# sub-hc002,  sub-hc009,  sub-hc019,  sub-hc024,  sub-hc031,   sub-tbi003,  sub-tbi008,  sub-tbi013,  sub-tbi018,  sub-tbi024,  sub-tbi034,
# sub-hc003,  sub-hc010,  sub-hc020,  sub-hc025,  sub-hc033,   sub-tbi004,  sub-tbi009,  sub-tbi014,  sub-tbi019,  sub-tbi025,  sub-tbi035,
# sub-hc004,  sub-hc011,  sub-hc021,  sub-hc028,  sub-hc034,   sub-tbi005,  sub-tbi010,  sub-tbi015,  sub-tbi020,  sub-tbi027,  sub-tbi036,
# sub-hc005,  sub-hc012,  sub-hc022,  sub-hc029,  sub-tbi001,  sub-tbi006,  sub-tbi011,  sub-tbi016,  sub-tbi022,  sub-tbi029]
params = edict({
        'path': '/data/fmri',
        'nGPU': 2,
        'nEpochs': 10,
        'nBacks': 2,
        'nTasks': 4,
        'nClass': 6,
        'batchSize': 10,
        'nT': 135,
        'nX': 57,
        'nY': 68,
        'nZ': 49,
        'nDivT': 9,
        'ndf': 16,
        'lr': 0.001,
        'beta1': 0.5,
        'beta2': 0.999
})

def get_counts(test_pct):
    hc_subs = [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 16, 19, 20, 21, 22, 23, 24, 25, 28, 29, 30, 31, 33]
    tbi_subs = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 27, 29, 30, 34, 35, 36]
    
    test_hc_count = int(len(hc_subs)*test_pct)
    test_tbi_count = int(len(tbi_subs)*test_pct)
    hc_test = random.sample(hc_subs, test_hc_count)
    tbi_test = random.sample(tbi_subs, test_tbi_count)
    hc_train = list(set(hc_subs) - set(hc_test))
    tbi_train = list(set(tbi_subs) - set(tbi_test))
    
    return hc_train, hc_test, tbi_train, tbi_test

# def get_counts(test_pct):
#     hc_subs = [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 16, 19, 20, 21, 22, 23, 24, 25, 28, 29, 30, 31, 33]
#     tbi_subs = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 27, 29, 30, 34, 35, 36]
#     total_subs = len(hc_subs) + len(tbi_subs)
#     test_count = int(total_subs*test_pct)
#     train_count = total_subs - test_count
#     # Randomly sample subjects for train and test sets
#     hc_train = random.sample(hc_subs, train_count//2)
#     hc_test = list(set(hc_subs) - set(hc_train))
#     tbi_train = random.sample(tbi_subs, (train_count-train_count//2))
#     tbi_test = list(set(tbi_subs) - set(tbi_train))
#     return hc_train, hc_test, tbi_train, tbi_test

def train_test_subs(test_pct=0.2):
    hc_train, hc_test, tbi_train, tbi_test = get_counts(test_pct)
    train_subs, test_subs = [], []
    for tr in hc_train:
        train_subs.append(f'sub-hc{tr:03}')
    for tr in tbi_train:
        train_subs.append(f'sub-tbi{tr:03}')
    for tst in hc_test:
        test_subs.append(f'sub-hc{tst:03}')
    for tst in tbi_test:
        test_subs.append(f'sub-tbi{tst:03}')

    return train_subs, test_subs