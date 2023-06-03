from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import plots as pl
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


teamsOriginal = np.load("data.npy", allow_pickle=True)
labels = np.load("nayve.npy", allow_pickle=True)

print(labels.shape)

teams = teamsOriginal[ : , : , 1: ]
teams = teams.astype(np.float32)

isPCA = True

if isPCA:
    teams = PCA(n_components=2).fit_transform(teams.reshape(-1,7)).reshape(30,16,-1)


def split(x,y):
    y = y.astype(np.int64)
    return x[:21] , y[:21] , x[21:],y[21:]

Xtrain, ytrain, Xtest, ytest = split(teams , labels.T)


print(ytrain.shape)
print(Xtrain.shape)

ytrain = np.transpose(ytrain)
ytrain = np.reshape(ytrain, -1)

Xtrain = np.reshape(Xtrain, (-1, teams.shape[-1]))
Xtest = np.reshape(Xtest, (-1, teams.shape[-1]))
teams = np.reshape(teams, (-1, teams.shape[-1]))


Nayve = GaussianNB()
Nayve.fit(Xtrain, ytrain)
predictions = Nayve.predict(Xtest)

ytest = ytest.ravel()
f1 = f1_score(ytest, predictions, average='macro')
accuracy = accuracy_score(ytest, predictions)
recall = recall_score(ytest, predictions, average='macro')
precision = precision_score(ytest, predictions, average='macro')


svm = SVC()
svm.fit(Xtrain, ytrain)
predictions = svm.predict(Xtest)


ytest = ytest.ravel()
f1 = f1_score(ytest, predictions, average='macro')
accuracy = accuracy_score(ytest, predictions)
recall = recall_score(ytest, predictions, average='macro')
precision = precision_score(ytest, predictions, average='macro')

print(f1, accuracy, recall, precision)

