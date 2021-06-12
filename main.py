import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable tensorflow warnings

from statistics import perform_wilcoxon_test

import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from nn import create_model
from util import load_data

import eli5
from eli5.sklearn import PermutationImportance


data = load_data()

Y = data['Ugięcia'].to_numpy()
#Y = np.array([0, 0, 1, 1])

X = data.drop('Ugięcia', axis='columns')
#print(Y.dtype)
#print(X.dtypes)

kNN_weights = ['uniform', 'distance']
kNN_neighbors = [1, 3, 5, 7, 10, 15, 20]
kNN_metric = ['euclidean', 'manhattan']

svr_param_C = [0.25, 0.5, 1, 2, 3, 4, 5]
svr_degree = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# svr_param_C = [0.25]
# svr_degree = [3]
#
# kNN_weights = ['uniform']
# kNN_neighbors = [5]
# kNN_metric = ['euclidean']

n_splits = 5
n_repeats = 2

cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=71)

kNN_scores = {}
svm_scores = {}
nn_scores = {}

# w tym słowniku będzie przechowywana informacja o najlepszej konfiguracji knn
best_kNN = {
    "best_value": -1,
    "samples": [],
    "weight": -1,
    "neighbor": -1,
    "metric": -1
}

best_SVM = {
    "best_value": -1,
    "samples": [],
    "c": -1,
    "degree": -1
}

for weight in kNN_weights:
    for neighbor in kNN_neighbors:
        for metric in kNN_metric:
            clf = KNeighborsRegressor(n_neighbors=neighbor, weights=weight, metric=metric)
            score = cross_val_score(clf, X, Y, scoring='neg_mean_squared_error', cv=cv)
            kNN_scores["w={0},k={1},m={2}".format(weight[0], neighbor, metric[0])] = score
            mean_of_scores = np.mean(score)
            if mean_of_scores > best_kNN["best_value"]:
                best_kNN["best_value"] = mean_of_scores
                best_kNN["samples"] = score
                best_kNN["weight"] = weight
                best_kNN["neighbor"] = neighbor
                best_kNN["metric"] = metric

#perform_ttest(list(kNN_scores.values()), list(kNN_scores.keys()))
# family wise error - skorygować p-value, aby skontrolować moc testu i współczynnik blędu
# Na stronie kursu folder z artykułami - sprawdzić!!

for c in svr_param_C:
    for degree in svr_degree:
        clf = SVR(C=c, degree=degree)
        score = cross_val_score(clf, X, Y, scoring='neg_mean_squared_error', cv=cv)
        svm_scores["SVM, c={0}, degree={1}".format(c, degree)] = score
        mean_of_scores = np.mean(score)
        if mean_of_scores > best_SVM["best_value"]:
            best_SVM["best_value"] = mean_of_scores
            best_SVM["samples"] = score
            best_SVM["c"] = c
            best_SVM["degree"] = degree

hidden_activations = ['sigmoid', 'relu', 'tanh']
hidden_layer_neurons = [1, 2, 5, 10, 15, 20, 25, 50, 100, 200, 500, 1000, 2500, 5000, 10000]
loss_functions = ['mean_squared_error', 'mean_absolute_error']

# hidden_activations = ['relu']
# hidden_layer_neurons = [200]
# loss_functions = ['mean_squared_error']

best_NN = {
    "best_value": -1,
    "samples": [],
    "hidden_activation": -1,
    "hidden_layer_neurons": -1,
    "loss_function": -1,
    "model" : None
}

nn_scores_arr = np.zeros((len(hidden_activations), len(hidden_layer_neurons), len(loss_functions), n_splits*n_repeats))

for i, hidden_activation in enumerate(hidden_activations):
    for j, hid_layer_neurons in enumerate(hidden_layer_neurons):
        for k, loss in enumerate(loss_functions):
            nn = create_model(X.shape[1], hidden_layer_neurons=hid_layer_neurons,
                              hidden_activation=hidden_activation, loss=loss)
            clf = KerasRegressor(build_fn=lambda: nn, epochs=25, verbose=0)
            score = cross_val_score(clf, X, Y, scoring='neg_mean_squared_error', cv=cv)
            nn_scores["NN, activation={0}, neurons={1}, loss={2}".format(hidden_activation, hid_layer_neurons, loss)] = score
            nn_scores_arr[i][j][k][:] = score
            mean_of_scores = np.mean(score)
            if mean_of_scores > best_NN["best_value"]:
                best_NN["best_value"] = mean_of_scores
                best_NN["samples"] = score
                best_NN["hidden_activation"] = hidden_activation
                best_NN["hidden_layer_neurons"] = hid_layer_neurons
                best_NN["loss_function"] = loss
                best_NN["model"] = clf

# Wypisanie najlepszych:
print(best_kNN)
print(best_SVM)
print(best_NN)

# Test Wilcoxona:
perform_wilcoxon_test(np.array([best_kNN["samples"], best_SVM["samples"], best_NN["samples"]]), ["kNN", "SVM", "NN"])

np.save('nn_scores', nn_scores_arr)

clf = best_NN["model"]
clf.fit(X, Y)
perm = PermutationImportance(clf, random_state=1).fit(X, Y) # https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html - opis jak to działa

with open('results.htm', 'wb') as f:
    f.write(eli5.show_weights(perm, feature_names=X.columns.tolist()).data.encode("UTF-8"))

print(kNN_scores)
print(svm_scores)
print(nn_scores)
