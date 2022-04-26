# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 20:50:29 2022

@author: Kuo yu cheng
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error

import keras
import sys

class Trade_Action():
    def __init__(self, stock):
        self.stock = stock              # stock status Own or Not
        self.action = []                # Save trade status (output)
        
        self.tomorrow_close_price = 0   # define tomorrow
        self.today_close_price = 0      # define today 

    def trade(self, pred_price):
        
        # define
        pr_close = pred_price

        for i in range(pred_price.shape[0] - 1):
            
            self.tomorrow_close_price = pr_close[i + 1]     # tomorrow
            self.today_close_price = pr_close[i]            # today

            if i == 0:                  # ingore the first day
                self.action.append(0)
            else:
                # strong 
                if self.tomorrow_close_price > self.today_close_price:  
                    if self.stock == 0:
                        self.action.append(1)
                        self.stock += 1 
                    elif self.stock == 1:
                        self.action.append(-1)
                        self.stock -= 1
                    elif self.stock == -1:
                        self.action.append(0)
                        self.stock += 0
                    else:
                        sys.exit("Trading status error")
                # unchanged
                elif self.tomorrow_close_price == self.today_close_price:
                        self.action.append(0)
                        self.stock += 0
                # weak
                else:
                    if self.stock == 0:
                        self.action.append(-1)
                        self.stock -= 1
                    elif self.stock == 1:
                        self.action.append(0)
                        self.stock += 0
                    elif self.stock == -1:
                        self.action.append(1)
                        self.stock += 1
                    else:
                        sys.exit("Trading status error")
        return self.action

def LSTM_model(X_train,y_train):
    keras.backend.clear_session()
    LSTM_model = Sequential()
    LSTM_model.add(LSTM(units = 16, batch_input_shape = (1 ,X_train.shape[1], 1), 
                        stateful= True))
    LSTM_model.add(Dense(units = 1))
    LSTM_model.summary()

    LSTM_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    early_stop = keras.callbacks.EarlyStopping( monitor="loss")
    LSTM_model.fit(X_train, y_train, epochs = 20, batch_size = 1, callbacks=[early_stop])
    return LSTM_model

class preprocessing():    
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def data_load(self):
        self.train = pd.read_csv(self.train, header=None)
        self.test = pd.read_csv(self.test, header = None)
        self.train = pd.concat([self.train,self.test], axis = 0)
        return self.train, self.test

    def select_feature(self):
        self.train_set = self.train.iloc[:,[3]]  # choose close as feature
        self.test_set = self.test.iloc[:,[3]]    # choose close as feature
        return self.train_set, self.test_set

    def data_scaler(self, scaler, prices, inverse):
        if inverse is False:
            self.training_set_scaled = scaler.fit_transform(self.train_set)
            return self.training_set_scaled
        else:
            return scaler.inverse_transform(prices)

    def build_train_data(self, shift):
        self.training_data = self.training_set_scaled[0:-20]
        X_train = []
        y_train = []

        for i in range(shift, len(self.training_data), 1):
            X_train.append(self.training_data[i-shift: i])
            y_train.append(self.training_data[i])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        y_train = np.reshape(y_train, (y_train.shape[0], 1))

        return X_train, y_train
    
    def build_test_data(self, shift):
        self.testing_data = self.training_set_scaled[-35:]
        X_test = []
        y_test = []

        for i in range(shift, len(self.testing_data)):
            X_test.append(self.testing_data[i-shift: i])
            y_test.append(self.testing_data[i])
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        y_test = np.reshape(y_test, (y_test.shape[0], 1))

        return X_test, y_test

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()


                                     
    sc = MinMaxScaler(feature_range = (0, 1))       
    training_shift = 15                             # path 15 days predict the next day

    # Load Data
    load_data = preprocessing(args.training, args.testing)
    all_data, test_data = load_data.data_load()
    train_set, test_set = load_data.select_feature()
    train_set_sc = load_data.data_scaler(sc, 0, inverse = False)
    X_train, y_train = load_data.build_train_data(training_shift)
    X_test, y_test = load_data.build_test_data(training_shift)

    # Predict result
    predicted_stock_price = LSTM_model(X_train,y_train).predict(X_test, batch_size = 1)
    predicted_stock_price = load_data.data_scaler(sc, predicted_stock_price, inverse = True)
    
    # MSE evaluation
    stock_true = test_set
    stock_pred = predicted_stock_price
    MSE = mean_squared_error(stock_true, stock_pred)

    print("MSE = ", MSE)

    Trading_act = Trade_Action(stock = 0)
    action = Trading_act.trade(predicted_stock_price)


    
    # plot all_data High Low Close
    plt.figure(figsize=(10,8))
    np_all_data = np.array(all_data) 
    plt.plot(np_all_data[:,1], color='red', label='High')
    plt.plot(np_all_data[:,2], color='green', label='Low')
    plt.plot(np_all_data[:,3], '--', color='black', label='Close', linewidth=2)
    plt.title('Stock Price Curve')
    plt.ylabel("Price")
    plt.xlabel("Dates")
    plt.legend()
    plt.savefig("Stock Price Curve.png")

    # plot Predict & Ans
    plt.figure(figsize=(10,8))
    plt.title("Stock Close")
    plt.plot(predicted_stock_price, color = 'red', label = 'Predict')
    plt.plot(stock_true, color = 'blue', label = 'Ans')
    plt.ylabel("Price")
    plt.xlabel("Dates")
    plt.legend()
    plt.tight_layout()

    plt.savefig("Stock Close Result.png")
    plt.show()
    

    with open(args.output, 'w') as output_file:
        for i in range(len(action)):
            output_file.writelines(str(action[i])+"\n")