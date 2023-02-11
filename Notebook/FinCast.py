# import the necessary libraries
import yfinance as yf
from pandas_datareader import data as pdr
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime as dt

#  import tensorflow libraries
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import os 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# getting latest data
def get_data(stocks, start, end):
    df = pdr.get_data_yahoo(stocks, start, end)
    return df

endDate = datetime.datetime.now()
startDate = endDate - datetime.timedelta(days=30)

def lstm_split(data, n_steps): 
  X,y =[], []
  for i in range(len(data)-n_steps+1):
    X.append(data[i:i+n_steps, :-1])
    y.append(data[i+n_steps-1,-1])
  return np.array(X), np.array(y)

def plotGraph(y_test,y_pred,regressorName):
    if max(y_test) >= max(y_pred):
        my_range = int(max(y_test))
    else:
        my_range = int(max(y_pred))
    plt.figure(figsize=(15,10))
    plt.plot(range(len(y_test)), y_test, color='blue')
    plt.plot(range(len(y_pred)), y_pred, color='red')
    plt.legend()
    plt.title(regressorName)
    plt.show()
    return
  
  def LSTM_model(X_train):
    
    model = Sequential()
    
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=1))
    
    return model
  
  # We try with another model to see if we get better predictions

for tick in tickers:
  prediction_days = 60
  scaler = MinMaxScaler(feature_range=(0,1))
  START_DATE = dt.datetime(2015,1,1)
  END_DATE = dt.datetime(2023,1,1)
  df = get_data(tick, START_DATE, END_DATE)
  scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))
  x_train = []
  y_train = []

  for x in range(prediction_days, len(scaled_data)):
      x_train.append(scaled_data[x - prediction_days:x, 0])
      y_train.append(scaled_data[x, 0])
  print(f"Data Preprocessing for {tick} Stocks")
  x_train, y_train = np.array(x_train), np.array(y_train)
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
  lstm = LSTM_model(x_train)
  lstm.summary()
  lstm.compile(optimizer='adam', 
                loss='mean_squared_error')
  checkpointer = ModelCheckpoint(filepath = tick+'_weights_best.hdf5', 
                               verbose = 2, 
                               save_best_only = True)

  lstm.fit(x_train, 
            y_train, 
            epochs=25, 
            batch_size = 32,
            callbacks = [checkpointer])
  END_DATE = dt.datetime(2023,1,1)

  START_DATE_TEST = END_DATE
  test_data = get_data(tick, START_DATE_TEST,dt.datetime.now())
  actual_prices = test_data['Close'].values

  total_dataset = pd.concat((df['Close'], test_data['Close']), axis=0)

  lstm_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
  lstm_inputs = lstm_inputs.reshape(-1,1)
  lstm_inputs = scaler.transform(lstm_inputs)
  x_test = []
  for x in range(prediction_days, len(lstm_inputs)):
      x_test.append(lstm_inputs[x-prediction_days:x, 0])

  x_test = np.array(x_test)
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] ,1))

  predicted_prices = lstm.predict(x_test)
  predicted_prices = scaler.inverse_transform(predicted_prices)
    # save model 
  directory = "lstm_saved_model"
  parent_dir = "/content"
  path = os.path.join(parent_dir, directory)
  os.makedirs(path, exist_ok=True) 
  print("Directory '% s' created" % directory) 

  lstm.save('lstm_saved_model/'+tick+'_model')
  plt.figure(figsize=(15,10))
  plt.plot(actual_prices, color='blue', label=f"Actual {tick} price")
  plt.plot(predicted_prices, color= 'green', label=f"Predicted {tick} price")
  plt.title(f"{tick} share price")
  plt.xlabel("time")
  plt.ylabel(f"{tick} share price")
  plt.legend()
  plt.show()
  # predicting next day
  real_data = [lstm_inputs[len(lstm_inputs)+1 - prediction_days:len(lstm_inputs+1),0]]
  real_data = np.array(real_data)
  real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
  prediction = lstm.predict(real_data)
  prediction = scaler.inverse_transform(prediction)
  print(f"prediction: {prediction}")

  
