from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from nn import create_model
from scores import cross_val, predict, fit_and_predict
from util import load_data, transform_data, stratified_train_test_split, random_train_test_split
from pprint import pprint

data = load_data()
X_train, X_test, y_train, y_test = stratified_train_test_split(data)

X_train = transform_data(X_train)
X_test = transform_data(X_test)

kNN_params_grid = [
    {
        'weights': ['uniform', 'distance'],
        'n_neighbors': [1, 3, 5, 7, 10, 15, 20],
        'metric': ['euclidean', 'manhattan']
    }
]
#
grid_search = GridSearchCV(KNeighborsRegressor(), kNN_params_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train, y_train)
#pprint(grid_search.cv_results_)
print(grid_search.best_params_)
print(grid_search.best_score_)

print(predict(grid_search.best_estimator_, X_test, y_test))

svr_params_grid = [
    {
        'C': [0.25, 0.5, 1, 2, 3, 4, 5],
        'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
]

grid_search = GridSearchCV(SVR(kernel='poly'), svr_params_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)

print(predict(grid_search.best_estimator_, X_test, y_test))

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
            nn = create_model(X_train.shape[1], hidden_layer_neurons=hid_layer_neurons, hidden_activation=hidden_activation, loss=loss)
            nn.fit(X_train, y_train, epochs=25, verbose=0)
            mse = predict(nn, X_test, y_test)
            if mse < best:
                best = mse
                best_nn = hidden_activation + ", " + str(hid_layer_neurons) + ", " + loss
            print('\t\tloss function: '+ loss)
            print('\t\t\t MSE: ' + str(mse))

print(best_nn, ",", best)