import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math, pickle

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class RNN_LSTM_Model():

    def __init__(self, dataset: str, epochs: int, datetime_column_name: str, data_column_name: str, steps_ahead: int, prediction_steps: int):

        self.dataset = dataset
        self.epochs = epochs
        self.datetime_column_name = datetime_column_name
        self.data_column_name = data_column_name
        self.steps_ahead = steps_ahead
        self.prediction_steps = prediction_steps

        self.train_size = 0
        self.test_size = 0
        self.sc = None

    def dataset_preparation(self):

        # dataset
        df = pd.read_csv(self.dataset)

        # order by Datatime column ascending
        df = df.sort_values(by=[self.datetime_column_name], ascending=True)

        # set 'Datetime' column as index of dataset
        df.Datetime = pd.to_datetime(df.Datetime)
        df = df.set_index(self.datetime_column_name)

        return df

    def dataset_split(self):

        day_mean = self.dataset_preparation()

        # split data in train and test
        self.train_size = int(len(day_mean) * 0.75)
        self.test_size = len(day_mean) - self.train_size

        train = day_mean[:self.train_size]
        test = day_mean[self.train_size:]

        return train, test

    def data_processing(self):

        # split data in train and test
        train, test = self.dataset_split()

        # normalize data
        self.sc = MinMaxScaler(feature_range = (0, 1))
        train_scaled = self.sc.fit_transform(train)

        return train_scaled

    def LSTM_dataset_shape(self):

        train_scaled = self.data_processing()

        # shape data
        X_train = []
        y_train = []

        for i in range(self.steps_ahead, self.train_size):
            X_train.append(train_scaled[i-self.steps_ahead:i, 0])
            y_train.append(train_scaled[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        return X_train, y_train

    # make predictions n steps ahead
    def moving_test_window_preds(self, X_test, regressor, n_feature_preds):

        n_feature_preds = n_feature_preds + 1

        preds_moving=[]
        moving_test_window=[X_test[X_test.shape[0] - 1,:].tolist()]
        moving_test_window=np.array(moving_test_window)
    
        for i in range(n_feature_preds):
            preds_one_step=regressor.predict(moving_test_window)
            preds_moving.append(preds_one_step[0,0])
            preds_one_step=preds_one_step.reshape(1,1,1)
            moving_test_window=np.concatenate((moving_test_window[:,1:,:],preds_one_step), axis= 1)

        preds_moving=np.array(preds_moving)
        preds_moving.reshape(1,-1)
        preds_moving = self.sc.inverse_transform(preds_moving.reshape(-1,1))
        
        # final list with predictions
        final_preds = []

        for i in preds_moving:
            final_preds.append(i)

        final_preds.pop(0) 

        return final_preds

    def train_model(self):

        X_train, y_train = self.LSTM_dataset_shape()

        # LSTM model        
        regressor = keras.Sequential()

        regressor.add(keras.layers.LSTM(units = 128, activation='tanh', return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(keras.layers.Dropout(rate=0.2))

        regressor.add(keras.layers.LSTM(units = 128, activation='tanh', return_sequences = True))
        regressor.add(keras.layers.Dropout(rate=0.2))

        regressor.add(keras.layers.LSTM(units = 128, activation='tanh'))
        regressor.add(keras.layers.Dropout(rate=0.2))
        
        regressor.add(keras.layers.Dense(units = 1, activation='tanh'))

        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        
        # train the model
        regressor.fit(X_train, y_train, epochs=self.epochs, batch_size=64, shuffle=False, validation_split=0.1)
        
        # save the model
        regressor.save('test2.h5')

        return 1

    def forecast(self, n_feature_preds):
        
        # data processing
        self.data_processing()

        # dataset splited in train and test
        train, test = self.dataset_split()

        # trained model
        regressor = keras.models.load_model('test.h5')

        # normalizing data
        dataset_total = pd.concat((train[self.data_column_name], test[self.data_column_name]), axis = 0)
        inputs = dataset_total[len(dataset_total) - len(test) - self.steps_ahead:].values
        inputs = inputs.reshape(-1,1)
        inputs = self.sc.transform(inputs)

        # put dataset in the LSTM format
        X_test = []
        for i in range(self.steps_ahead, self.test_size + self.steps_ahead):
            X_test.append(inputs[i-self.steps_ahead:i, 0])

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # predictions
        predictions = self.moving_test_window_preds(X_test, regressor, n_feature_preds)

        return predictions