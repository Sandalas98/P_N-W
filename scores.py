from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np


def cross_val(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train,
                             scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-scores)
    return rmse_scores.mean(), rmse_scores.std()


def predict(model, X_test, y_test):
    predictions = model.predict(X_test)
    lin_mse = mean_squared_error(y_test, predictions)
    lin_rmse = np.sqrt(lin_mse)
    return lin_rmse


def fit_and_predict(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    lin_mse = mean_squared_error(y_test, predictions)
    lin_rmse = np.sqrt(lin_mse)
    return lin_rmse