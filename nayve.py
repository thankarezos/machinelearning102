import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import pandas as pd


teamsOriginal = np.load("data.npy", allow_pickle=True)
labels = np.load("nayve.npy", allow_pickle=True)

teams = teamsOriginal[ : , : , 1: ]
teams = teams.astype(np.float32)

isPCA = True

if isPCA:
    teams = PCA(n_components=2).fit_transform(teams.reshape(-1,7)).reshape(30,16,-1)


def split(x,y):
    y = y.astype(np.int64)
    return x[:21] , y[:21] , x[21:],y[21:]

Xtrain, ytrain, Xtest, ytest = split(teams , labels.T)

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
precision = precision_score(ytest, predictions, average='macro', zero_division=1)

vayneMetrcis = pd.DataFrame([f1, accuracy, recall, precision], columns=['Nayve'], index=['f1', 'accuracy', 'recall', 'precision'])

svm = SVC()
svm.fit(Xtrain, ytrain)
predictions = svm.predict(Xtest)


ytest = ytest.ravel()
f1 = f1_score(ytest, predictions, average='macro')
accuracy = accuracy_score(ytest, predictions)
recall = recall_score(ytest, predictions, average='macro')
precision = precision_score(ytest, predictions, average='macro', zero_division=1)

vayneMetrcis["SVM"] = [f1, accuracy, recall, precision]
print(vayneMetrcis)

