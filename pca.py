from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf
import pandas as pd
import plots as pl

def pca():
    teamsOriginal = np.load("data.npy", allow_pickle=True)
    labels = np.load("labels.npy", allow_pickle=True)
    labels -= 1

    teams = teamsOriginal[ : , : , 1: ]
    teams = teams.astype(np.float32)
    teams = PCA(n_components=2).fit_transform(teams.reshape(-1,7)).reshape(30,16,-1)
    teams = tf.convert_to_tensor(teams)

    def model():
        input = tf.keras.layers.Input((2))
        x = tf.keras.layers.Dense(512,'relu')(input)
        x = tf.keras.layers.Dense(54,'relu')(x)
        x= tf.keras.layers.Dense(6,'softmax')(x)
        return tf.keras.models.Model(inputs= input , outputs = x)

    Model = model()

    def split(x,y):
        y = y.astype(np.int64)
        y = tf.convert_to_tensor(y)
        return x[:21] , y[:21] , x[21:],y[21:]

    Xtrain, ytrain, Xtest, ytest = split(teams , labels.T)

    Model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    ytrain = tf.transpose(ytrain)
    ytrain = tf.reshape(ytrain, -1)

    Xtrain = tf.reshape(Xtrain, (-1, 2))
    Xtest = tf.reshape(Xtest, (-1, 2))
    teams = tf.reshape(teams, (-1, 2))
    labels = labels.astype(float)
    teamsLabels = tf.convert_to_tensor(labels)
    teamsLabels = tf.reshape(labels, (-1, 1))

    Model.fit(Xtrain, ytrain, batch_size=6, epochs=100, verbose=1)
    predictions = Model(teams)

    predictions = predictions.numpy().argmax(axis=-1).reshape(30, 16)

    print("accuracy: ", Model.evaluate(teams, teamsLabels)[1])




    df = pd.read_excel('data.xlsx')
    data = df.iloc[:8, :5]
    data.iloc[:, 1:3] = data.iloc[:, 1:3].apply(lambda x: x.str.strip())
    teamNames = pd.concat([data.iloc[:, 1], data.iloc[:, 2]], ignore_index=True)

    score = [0] * len(teamNames)

    teams = pd.DataFrame({'Name': teamNames})


    for i in range(0, 30):

        column_name = f'{i + 1}'
        teams[column_name] = predictions[i] + 1

    print(teams)
    return teams