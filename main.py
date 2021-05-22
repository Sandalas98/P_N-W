import numpy as np
import pandas as pd
import zipfile
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.read_csv("data/data.csv", sep=';')

for column in data.columns:
    if data[column].dtype != np.float64 or data[column].dtype != np.int64:
        data[column] = data[column].astype('category')
        data[column] = data[column].cat.codes

classifieres_names = ['kNN', 'SVM']
classifieres = [
    KNeighborsClassifier(5),
    SVC()]


target = data['Obciążenie']
data_features = data.drop(['Obciążenie'], axis='columns')

max_abs_scaler = MaxAbsScaler()

data_features_scaled = max_abs_scaler.fit_transform(data_features)

data_features_scaled = pd.DataFrame(
    data_features_scaled, columns=data_features.columns)


data_train_X, data_test_X, data_train_Y, data_test_Y = train_test_split(
    data_features_scaled, target, test_size=0.2, random_state=71)


clf = classifieres[0]
res = clf.fit(data_train_X, data_train_Y)
predicted = clf.predict(data_test_X)

print(predicted)
print(data_test_Y)

acc = accuracy_score(data_test_Y, predicted)
print(acc)

#for classifier, classifier_name in classifieres, classifieres_names:
    #print(classifieres_names[classifier_name])
    #classifieres[classifier].fit(data_train_X, data_train_Y)