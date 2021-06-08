import numpy as np
from pip._internal.utils.misc import tabulate
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import rankdata
from sklearn.svm import SVR
from scipy.stats import wilcoxon

from nn import create_model
from scores import cross_val, predict, fit_and_predict
from util import load_data, transform_data, stratified_train_test_split, random_train_test_split
from pprint import pprint

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

cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=71)

kNN_scores = []
svm_scores = []
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
            score = cross_val_score(clf, X, Y, scoring='neg_mean_absolute_error', cv=cv)
            mean_of_scores = np.mean(score)
            if mean_of_scores > best_kNN["best_value"]:
                best_kNN["best_value"] = mean_of_scores
                best_kNN["samples"] = score
                best_kNN["weight"] = weight
                best_kNN["neighbor"] = neighbor
                best_kNN["metric"] = metric




# family wise error - skorygować p-value, aby skontrolować moc testu i współczynnik blędu
# Na stronie kursu folder z artykułami - sprawdzić!!


for c in svr_param_C:
    for degree in svr_degree:
        clf = SVR(C=c, degree=degree)
        score = cross_val_score(clf, X, Y, scoring='neg_mean_absolute_error', cv=cv)
        mean_of_scores = np.mean(score)
        if mean_of_scores > best_SVM["best_value"]:
            best_SVM["best_value"] = mean_of_scores
            best_SVM["samples"] = score
            best_SVM["c"] = c
            best_SVM["degree"] = degree

        #svm_scores.append(score)

# Wypisanie najlepszych:
print(best_kNN)
print(best_SVM)
# Tutaj wchodzą sieci neuronowe:

# Test Wilcoxona:

kNN_data = best_kNN["samples"]
svm_data = best_SVM["samples"]

statistics, p_value = wilcoxon(kNN_data, svm_data)

print(statistics)
print(p_value)

if p_value > .05:
    print("Ten sam rozkład")
else:
    print("Różny rozkład")

'''
X_train, X_test, y_train, y_test = stratified_train_test_split(data)

X_train = transform_data(X_train)
X_test = transform_data(X_test)

#
grid_search = GridSearchCV(KNeighborsRegressor(), kNN_params_grid, cv=5, scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(X_train, y_train)
print("Grid search:")
print(grid_search)
# pprint(grid_search.cv_results_)
print("best params for KNN: ", grid_search.best_params_)
print("score: ", grid_search.best_score_)

print("result on test set: ", predict(grid_search.best_estimator_, X_test, y_test))


grid_search = GridSearchCV(SVR(kernel='poly'), svr_params_grid, cv=5, scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(X_train, y_train)
print("Grid search:")
print(grid_search)
print("best params for SVR: ", grid_search.best_params_)
print("score: ", grid_search.best_score_)

print("result on test set: ", predict(grid_search.best_estimator_, X_test, y_test))

# sieć neuronowa:


hidden_activations = ['sigmoid', 'relu', 'tanh']
hidden_layer_neurons = [1, 2, 5, 10, 15, 20, 25, 50, 100, 200, 500, 1000, 2500, 5000, 10000]
# hidden_layer_neurons = [10, 100, 500, 1000, 2500, 5000]
loss_functions = ['mean_squared_error', 'mean_absolute_error']

best = 1

for hidden_activation in hidden_activations:
    print('hidden activation: ' + hidden_activation)
    for hid_layer_neurons in hidden_layer_neurons:
        print('\t' + 'hidden layer neurons: ' + str(hid_layer_neurons))
        for loss in loss_functions:
            nn = create_model(X_train.shape[1], hidden_layer_neurons=hid_layer_neurons,
                              hidden_activation=hidden_activation, loss=loss)
            nn.fit(X_train, y_train, epochs=25, verbose=0)
            mse = predict(nn, X_test, y_test)
            if mse < best:
                best = mse
                best_nn = hidden_activation + ", " + str(hid_layer_neurons) + ", " + loss
                best_nn_model = nn
            print('\t\tloss function: ' + loss)
            print('\t\t\t MSE: ' + str(mse))

print('BEST PARAMETERS:')
print(best_nn + ", score: ", best)
print("result on test set: ", predict(best_nn_model, X_test, y_test))
'''