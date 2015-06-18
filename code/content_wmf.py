import sys
import time

import numpy as np
from scipy import weave

import batched_inv_joblib


def linear_surplus_confidence_matrix(B, alpha):
    # To construct the surplus confidence matrix, we need to operate only on
    # the nonzero elements.
    # This is not possible: S = alpha * B
    S = B.copy()
    S.data = alpha * S.data
    return S


def log_surplus_confidence_matrix(B, alpha, epsilon):
    # To construct the surplus confidence matrix, we need to operate only on
    # the nonzero elements.
    # This is not possible: S = alpha * np.log(1 + B / epsilon)
    S = B.copy()
    S.data = alpha * np.log(1 + S.data / epsilon)
    return S


def factorize(S, num_factors, X=None, vad=None, num_iters=10, init_std=0.01,
              lambda_U_reg=1e-5, lambda_V_reg=1e-5, lambda_W_reg=1e-5,
              dtype='float32', random_state=None, verbose=False,
              recompute_factors=batched_inv_joblib.recompute_factors_batched,
              *args, **kwargs):

    num_users, num_items = S.shape
    if X is not None:
        assert X.shape[0] == num_items
        assert np.all(X[:, -1] == 1)

    if verbose:
        print "precompute S^T and X^TX (if necessary)"
        start_time = time.time()

    ST = S.T.tocsr()
    if X is not None:
        n_feats = X.shape[1]
        R = np.eye(n_feats)
        R[n_feats - 1, n_feats - 1] = 0
        XTXpR = X.T.dot(X) + lambda_W_reg * R

    if verbose:
        print "  took %.3f seconds" % (time.time() - start_time)
        print "run ALS algorithm"
        start_time = time.time()

    if type(random_state) is int:
        np.random.seed(random_state)
    elif random_state is not None:
        np.random.setstate(random_state)

    U = np.random.randn(num_users, num_factors).astype(dtype) * init_std
    V = None  # no need to initialize V, it will be overwritten anyway
    if X is not None:
        W = np.random.randn(X.shape[1], num_factors).astype(dtype) * init_std
    else:
        W = None

    for i in xrange(num_iters):
        if verbose:
            print("Iteration %d:" % i)
            start_t = _write_and_time('\tUpdating item factors...')
        V = recompute_factors(U, ST, lambda_V_reg, W=W, X=X, dtype=dtype,
                              *args, **kwargs)
        if verbose:
            print('\r\tUpdating item factors: time=%.2f'
                  % (time.time() - start_t))

            start_t = _write_and_time('\tUpdating user factors...')
        U = recompute_factors(V, S, lambda_U_reg, dtype=dtype, *args, **kwargs)

        if verbose:
            print('\r\tUpdating user factors: time=%.2f'
                  % (time.time() - start_t))
            pred_ll = _pred_loglikeli(U, V, dtype, **vad)
            print("\tPred likeli: %.5f" % pred_ll)
            sys.stdout.flush()

        if X is not None:
            if verbose:
                start_t = _write_and_time(
                    '\tUpdating projection matrix...')
            W = np.linalg.solve(XTXpR, X.T.dot(V))

            if verbose:
                print('\r\tUpdating projection matrix: time=%.2f'
                      % (time.time() - start_t))
    return U, V, W


def _pred_loglikeli(U, V, dtype, X_new=None, rows_new=None, cols_new=None):
    X_pred = _inner(U, V, rows_new, cols_new, dtype)
    pred_ll = np.mean((X_new - X_pred)**2)
    return pred_ll


def _write_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()


def _inner(U, V, rows, cols, dtype):
    n_ratings = rows.size
    n_components = U.shape[1]
    assert V.shape[1] == n_components
    data = np.empty(n_ratings, dtype=dtype)
    code = r"""
    for (int i = 0; i < n_ratings; i++) {
       data[i] = 0.0;
       for (int j = 0; j < n_components; j++) {
           data[i] += U[rows[i] * n_components + j] * V[cols[i] * n_components + j];
       }
    }
    """
    weave.inline(code, ['data', 'U', 'V', 'rows', 'cols', 'n_ratings',
                        'n_components'])
    return data
