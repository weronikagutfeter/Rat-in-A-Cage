
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
import pickle
import pandas as pd

from sktime.datasets import load_arrow_head, load_basic_motions, load_italy_power_demand

from sktime.transformations.panel.rocket import Rocket

from sktime.datatypes._panel._convert import (
    from_2d_array_to_nested,
    from_nested_to_2d_array,
    is_nested_dataframe,
_concat_nested_arrays,
from_long_to_nested
)

def convert2nested(X,series_len=30, stride=30):
    n_timeseries = len(X)
    n_dims = X[0].shape[1]

    total_rows = n_timeseries * series_len * 500 * n_dims

    ts_ids = np.empty(total_rows, dtype=int)
    obs_ids = np.empty(total_rows, dtype=int)
    dim_ids = np.empty(total_rows, dtype=int)
    values = np.empty(total_rows)

    src_idxs = []

    row_id = 0
    dest_ts_id = 0
    for src_ts_id in range(len(X)):
        timeserie = X[src_ts_id]
        for start_pos in range(0,len(timeserie)-series_len,stride):
            timeserie_part = timeserie[start_pos:start_pos+series_len,:]
            for observation_id in range(len(timeserie_part)):
                for dim_id in range(n_dims):
                    ts_ids[row_id] = dest_ts_id
                    dim_ids[row_id] = dim_id
                    obs_ids[row_id] = observation_id
                    values[row_id] = timeserie_part[observation_id,dim_id]
                    row_id += 1
            src_idxs.append(src_ts_id)
            dest_ts_id += 1

    ts_ids = ts_ids[:row_id]
    obs_ids = obs_ids[:row_id]
    dim_ids = dim_ids[:row_id]
    values = values[:row_id]


    df = pd.DataFrame()
    df["case_id"] = pd.Series(ts_ids)
    df["dim_id"] = pd.Series(dim_ids)
    df["reading_id"] = pd.Series(obs_ids)
    df["value"] = pd.Series(values)

    X_nested = from_long_to_nested(df)
    return X_nested, src_idxs



def main(datafile):
    with open(datafile, 'rb') as f:
        train_sessions,test_sessions = pickle.load(f)

        X_train = [s['data'] for s in train_sessions]
        X_test = [s['data'] for s in test_sessions]

        for binary_classification in  [False, True]:



            if (binary_classification):
                y_train = [int(s["catid"] > 0) for s in train_sessions]
                y_test = [int(s["catid"] > 0) for s in test_sessions]
            else:
                y_train = [s["catid"] for s in train_sessions]
                y_test = [s["catid"] for s in test_sessions]


            for series_len in [200]:#[100, 50, 30]:


                X_train_nested, ts_train_idxs = convert2nested(X_train, series_len=series_len, stride=100)
                X_test_nested, ts_test_idxs = convert2nested(X_test, series_len=series_len, stride=100)
                y_train_nested = [y_train[tid] for tid in ts_train_idxs]
                y_test_nested = [y_test[tid] for tid in ts_test_idxs]

                cats = np.unique(y_train_nested)

                print('Size of TRAINING DATASET: {} with {} classes'.format(len(y_train_nested),len(cats)))
                for k in cats:
                    print('{} samples with label {}'.format(np.sum(np.array(y_train_nested)==k),k))
                print('Size of TEST DATASET: {}'.format(len(y_test_nested)))
                for k in cats:
                    print('{} samples with label {}'.format(np.sum(np.array(y_test_nested)==k),k))

                rocket = Rocket()  # by default, ROCKET uses 10,000 kernels
                rocket.fit(X_train_nested)
                X_train_transform = rocket.transform(X_train_nested)

                classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))#, normalize=True)
                classifier.fit(X_train_transform, y_train_nested)

                X_test_transform = rocket.transform(X_test_nested)
                s = classifier.score(X_test_transform, y_test_nested)
                print('Classifier score for series_len={} and {} classes is {:.2f}'.format(series_len,len(cats),100*s))

                y_pred = classifier.predict(X_test_transform)

                print("Pred class \t| ",end='')
                for cat in cats:
                    print('{} \t|'.format(cat),end='')
                print()
                y_test_src = np.array([test_sessions[tid]["catid"] for tid in ts_test_idxs])
                src_cats = np.unique(y_test_src)
                for src_cat in src_cats:
                    print('{} \t'.format(src_cat),end='')
                    for cat in cats:
                        s = np.sum((y_pred==cat) & (y_test_src==src_cat))
                        print('{} \t|'.format(s), end='')
                    print()


if __name__ == '__main__':


    datafile = '../../sample_data/folds/Screen_ts_fold0.pkl'
    main(datafile)