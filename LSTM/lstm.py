from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.layers import LSTM
from keras import callbacks
from keras import optimizers
import pandas as pd
import tensorflow as tf
import numpy as np
import math
from sklearn.metrics import mean_squared_error


def read_df():
    df = pd.read_csv('data.csv')
    df = df.sort_values('unix')
    return df


def create_dataset(df):
    dataset = df
    dataset = dataset - dataset.shift(1)
    dataset = dataset.dropna()
    return dataset


def scale(dataset):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(dataset)
    print('Min', np.min(scaled))
    print('Max', np.max(scaled))
    return scaler, scaled


def train_test(scaled):
    train_size = int(len(scaled) * 0.70)
    test_size = len(scaled - train_size)
    train, test = scaled[0:train_size, :], scaled[train_size: len(scaled), :]
    print('train: {}\ntest: {}'.format(len(train), len(test)))
    return train, test


def create_dataset(dataset, look_back=1):
    print(len(dataset), look_back)
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        print(i)
        print('X {} to {}'.format(i, i+look_back))
        print(a)
        print('Y {}'.format(i + look_back))
        print(dataset[i + look_back, 0])
        dataset[i + look_back, 0]
        dataX.append(a)
        dataY.append(dataset[i:(i + look_back), 0])
    return np.array(dataX), np.array(dataY)


def x_train_test(train, test, look_back):
    look_back = 10
    X_train, y_train = create_dataset(train, look_back)
    X_test, y_test = create_dataset(test, look_back)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    print(X_train.shape)
    print(X_test.shape)
    return X_train, y_train, X_test, y_train


def trian(X_train, y_train, look_back):
    batch_size = 1
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=100, batch_size=batch_size, verbose=2, shuffle=True)
    return model


def predict_train(X_train, batch_size, model, scaler):
    trainPredict = model.predict(X_train, batch_size=batch_size)
    trainPredict = scaler.inverse_transform(trainPredict)
    y_train = scaler.inverse_transform(y_train)
    model.reset_states()
    return trainPredict


def predict_test(X_test, batch_size, model, scaler):
    curr_X = X_test[0]
    preds = []
    for i in range(len(X_test)):
        testPredict = model.predict(np.array([curr_X]), batch_size=batch_size)
        preds.append(testPredict[0])
        curr_X = list(curr_X[1:]) + list(testPredict)

    preds = np.array(preds)
    return preds


def metrics(y_test, testPredict, trainPredict, scaled, scaler):
    testScore = math.sqrt(mean_squared_error(y_test[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))
    trainPredictPlot = np.empty_like(scaled)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(scaled)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(scaled)-1, :] = testPredict
    # plot baseline and predictions
    plt.figure(figsize=(20, 10))
    plt.plot(scaler.inverse_transform(scaled))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
