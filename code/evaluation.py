import os
os.environ['OMP_NUM_THREADS'] = '10'
import sys
import time

import bottleneck as bn
import numpy as np
import scipy.sparse
import pandas as pd

import rec_eval

# DEN
n_users = 613682
n_songs = 92543

# SPR
#n_users = 564437
#n_songs = 260345


def load_data(csv_file, shape=(n_users, n_songs)):
    tp = pd.read_csv(csv_file)
    rows, cols = np.array(tp['uid'], dtype=np.int32), np.array(tp['sid'], dtype=np.int32)
    count = tp['count']
    return scipy.sparse.csr_matrix((count,(rows, cols)), dtype=np.int16, shape=shape), rows, cols


train_data, rows, cols = load_data('in.train.num.csv')
# binarize the data
train_data.data = np.ones_like(train_data.data)

vad_data, rows_vad, cols_vad = load_data('in.vad.num.csv')
# binarize the data
vad_data.data = np.ones_like(vad_data.data)

#test_data, rows_test, cols_test = load_data('in.test.num.csv')
test_data, rows_test, cols_test = load_data('in.test.num.unpop.csv')
# binarize the data
test_data.data = np.ones_like(test_data.data)

lam = float(sys.argv[1])
lamV = float(sys.argv[2])

params_wmf = np.load('params_wmf_K50_U%1.E_V%1.E.npz' % (lam, lam))
U_wmf, V_wmf = params_wmf['U'], params_wmf['V']

params_deep = np.load('params_deep_wmf_K50_U%1.E_V%1.E_W%1.E.npz' % (lam, lamV, lam))
U_deep, V_deep = params_deep['U'], params_deep['V']

params_shallow = np.load('params_shallow_wmf_K50_U%1.E_V%1.E_W%1.E.npz' % (lam, lamV, lam))
U_shallow, V_shallow = params_shallow['U'], params_shallow['V']

def recall_at_multiple_ks_batch(train_data, heldout_data, Et, Eb, user_idx,
                                topks, vad_data):
    batch_users = user_idx.stop - user_idx.start

    X_pred = rec_eval._make_prediction(train_data, Et, Eb, user_idx,
                                       batch_users, vad_data=vad_data)
    recalls = np.empty((len(topks), batch_users))
    for i, k in enumerate(topks):
        idx = bn.argpartsort(-X_pred, k, axis=1)
        X_pred_binary = np.zeros_like(X_pred, dtype=bool)
        X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

        X_true_binary = (heldout_data[user_idx] > 0).toarray()
        tmp = (np.logical_and(X_true_binary,
                              X_pred_binary).sum(axis=1)).astype(np.float32)
        recalls[i] = tmp / X_true_binary.sum(axis=1)
    return recalls


def recall_at_multiple_ks(train_data, heldout_data, U, V, batch_users=1000,
                          topks=range(40, 201, 40), vad_data=None):
    res = None
    start_t = time.time()
    for i, user_idx in enumerate(rec_eval.user_idx_generator(n_users,
                                                             batch_users), 1):
        if res is None:
            res = recall_at_multiple_ks_batch(train_data, test_data,
                                              U, V.T,
                                              user_idx, topks, vad_data)
        else:
            res = np.hstack((res,
                             recall_at_multiple_ks_batch(train_data, test_data,
                                                         U, V.T,
                                                         user_idx, topks,
                                                         vad_data)))
        sys.stdout.write('\rProgress: %d/%d\tTime: %.2f sec/batch' %
                         (user_idx.stop, n_users, (time.time() - start_t) / i))
        sys.stdout.flush()
    sys.stdout.write('\n')
    return np.nanmean(res, axis=1)


print("*****Recall@K*****")
print("WMF:")

recall_wmf = recall_at_multiple_ks(train_data, test_data, U_wmf, V_wmf, vad_data=vad_data)
print recall_wmf

print("deep:")

recall_deep = recall_at_multiple_ks(train_data, test_data, U_deep, V_deep, vad_data=vad_data)
print recall_deep

print("shallow:")

recall_shallow = recall_at_multiple_ks(train_data, test_data, U_shallow, V_shallow, vad_data=vad_data)
print recall_shallow

sys.stdout.flush()

print("\n*****NDCG*****")

ndcg_wmf = rec_eval.normalized_dcg(train_data, test_data, U_wmf, V_wmf, batch_users=1000, vad_data=vad_data)
print("WMF: %.5f" % ndcg_wmf)

ndcg_deep = rec_eval.normalized_dcg(train_data, test_data, U_deep, V_deep, batch_users=1000, vad_data=vad_data)
print("deep: %.5f" % ndcg_deep)

ndcg_shallow = rec_eval.normalized_dcg(train_data, test_data, U_shallow, V_shallow, batch_users=1000, vad_data=vad_data)
print("shallow: %.5f" % ndcg_shallow)

