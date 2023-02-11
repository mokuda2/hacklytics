import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd 
import datetime as dt
import matplotlib.pyplot as plt
def model_LSTM(tick):
    if tick == "AAPL":
        model = tf.keras.models.load_model('Models/lstm_saved_model/AAPL_model')
        print(model.summary())
    if tick =="TSLA":
        model = tf.keras.models.load_model('Models/lstm_saved_model/TSLA_model')
    if tick == "AMZN":
        model = tf.keras.models.load_model('Models/lstm_saved_model/AMZN_model')
    if tick == "GOOGL":
        model = tf.keras.models.load_model('Models/lstm_saved_model/GOOG_model')
    if tick == "MSFT":
        model = tf.keras.models.load_model('Models/lstm_saved_model/MSFT_model')
    if tick == "BTC-USD":
        model = tf.keras.models.load_model('Models/lstm_saved_model/BTC-USD_model')
    if tick == "ETH-USD":
        model = tf.keras.models.load_model('Models/lstm_saved_model/ETH-USD_model')
    if tick == "DOGE_USD":
        model = tf.keras.models.load_model('Models/lstm_saved_model/DOGE-USD_model')
    if tick == "XRP_USD":
        model = tf.keras.models.load_model('Models/lstm_saved_model/XRP-USD_model')
    return model 

def fit_LSTM(df, model):
    prediction_days = 60
    train_data, test_data = df[0:int(len(df)*0.7)], df[int(len(df)*0.7):]
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(train_data['Close'].values.reshape(-1,1))
    x_train = []
    y_train = []
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
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
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    real_data = [lstm_inputs[len(lstm_inputs)+1 - prediction_days:len(lstm_inputs+1),0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    prediction = model.predict(real_data)
    next_day_prediction = scaler.inverse_transform(prediction)
    return predicted_prices, actual_prices, next_day_prediction


def rolling_forecast_LSTM(ticker,df, predicted_prices, actual_prices, between_tick=20):
    test_set_range = df[int(len(df)*0.7):].index
    plt.figure(figsize=(20,10))
    plt.plot(test_set_range, actual_prices, color='red', label=f"Actual {ticker} price")
    plt.plot(test_set_range, predicted_prices, color= 'blue', marker='o', linestyle='dashed',label=f"Predicted {ticker} Price")
    plt.title(f"{ticker} Price Prediction")
    plt.xlabel("Date")
    plt.ylabel(f"Prices")
    plt.xticks(test_set_range[::between_tick], df.Date[0:len(actual_prices):between_tick], rotation=45)
    plt.legend()
    plt.savefig(f'graphs/LSTM.png', dpi=300, bbox_inches = 'tight')
