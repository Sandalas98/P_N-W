import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


def load_data():
    return pd.read_csv("data/data.csv", sep=';')


def transform_data(X):
    max_abs_scaler = MaxAbsScaler()
    X_scaled = max_abs_scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)


def stratified_train_test_split(data, test_size=0.2, random_state=42):
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    data["ugięcia_cat"] = pd.cut(data["Ugięcia"], bins=[-np.inf, 0.005, 0.01, 0.025, 0.05, np.inf],
                                 labels=[1, 2, 3, 4, 5])
    for train_index, test_index in split.split(data, data["ugięcia_cat"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
    strat_train_set = strat_train_set.drop("ugięcia_cat", axis=1)
    strat_test_set = strat_test_set.drop("ugięcia_cat", axis=1)

    X_train = strat_train_set.drop("Ugięcia", axis=1)
    y_train = strat_train_set["Ugięcia"].copy()
    X_test = strat_test_set.drop("Ugięcia", axis=1)
    y_test = strat_test_set["Ugięcia"].copy()
    return X_train, X_test, y_train, y_test


def random_train_test_split(data, test_size=0.2, random_state=42):
    y = data['Ugięcia']
    X = data.drop(['Ugięcia'], axis='columns')

    return train_test_split(X, y, test_size=test_size, random_state=random_state)
