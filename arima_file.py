import time
import math
import numpy as np 
import pandas as pd
import datetime
import scipy as sc 
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import yfinance as yf
yf.pdr_override()
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


def get_data(stocks, start, end, interval):
    df = pdr.get_data_yahoo(stocks, start, end, interval=interval)
    df = df.reset_index()
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    return df


def plot_stock(df, ticker, between_tick=10):
    df['Close'].plot(figsize=(20,5))
    plt.title(ticker)
    plt.ylabel('Price')
    set_range = df.index
    plt.xticks(set_range[::between_tick], df['Date'][::between_tick], rotation=45)
    plt.savefig('graphs/stock.png',  dpi=300, bbox_inches = 'tight')



def model_ARIMA(df):
    p_values = range(0,4)
    d_values = range(0, 4)
    q_values = range(0, 4)

    train_data, test_data = df[0:int(len(df)*0.7)], df[int(len(df)*0.7):]
    train_data = train_data['Close'].values
    test_data = test_data['Close'].values

    error_1 = np.inf 
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                warnings.filterwarnings("ignore")
                model = ARIMA(train_data, order=order).fit()
                predictions = model.predict(start=len(train_data), end=len(train_data) + len(test_data)-1)
                error_2 = mean_squared_error(test_data, predictions)
                if error_2 < error_1:
                    error_1 = error_2
                    best_order = order
    return best_order



def fit_ARIMA(df, p, d, q):

    train_data, test_data = df[0:int(len(df)*0.7)], df[int(len(df)*0.7):]
    training_data = train_data['Close'].values
    test_data = test_data['Close'].values
    history = [x for x in training_data]
    model_predictions = []
    N_test_observations = len(test_data)

    for time_point in range(N_test_observations):
        model = ARIMA(history, order=(p, d, q)) #change order here
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[time_point]
        history.append(true_test_value)

    return model_predictions, test_data
    


def rolling_forecast_ARIMA(ticker, df, model_predictions, test_data, between_tick=20):
    between_tick = 20
    test_set_range = df[int(len(df)*0.7):].index
    # set figure size
    plt.figure(figsize=(20,10))
    plt.plot(test_set_range, model_predictions, color='blue', marker='o', linestyle='dashed',label='Predicted Price')
    plt.plot(test_set_range, test_data, color='red', label='Actual Price')
    plt.title(f'{ticker} Prices Prediction')
    plt.xlabel('Date')
    plt.ylabel('Prices')
    plt.xticks(test_set_range[::between_tick], df.Date[0:len(test_data):between_tick], rotation=45)
    plt.legend()
    plt.savefig('graphs/ARIMA.png', dpi=300, bbox_inches = 'tight')



def find_order_d(df):
    
    df = df.drop(columns=['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'])
    fig, axes = plt.subplots(6, 2, sharex=True)
    fig.set_size_inches(15, 15)

    # Original Series
    axes[0, 0].plot(df); axes[0, 0].set_title('Original Series')
    plot_acf(df, ax=axes[0, 1], auto_ylims=True )
    adf = adfuller(df['Close'])
    axes[0, 1].text(0.1, 0.1, 'ADF p-value:'+str(round(adf[1],4)), {'fontsize': 20}, fontproperties = 'monospace')


    # 1st Differencing
    axes[1, 0].plot(df.diff()); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(df.diff().dropna(), ax=axes[1, 1], auto_ylims=True )
    adf = adfuller(df.diff().dropna()['Close'])
    axes[1, 1].text(0.1, 0.1, 'ADF p-value:'+str(round(adf[1],4)), {'fontsize': 20}, fontproperties = 'monospace')


    # 2nd Differencing
    axes[2, 0].plot(df.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(df.diff().diff().dropna(), ax=axes[2, 1], auto_ylims=True )
    adf = adfuller(df.diff().diff().dropna()['Close'])
    axes[2, 1].text(0.1, 0.1, 'ADF p-value:'+str(round(adf[1],4)), {'fontsize': 20}, fontproperties = 'monospace')

    # 3rd Differencing
    axes[3, 0].plot(df.diff().diff().diff()); axes[3, 0].set_title('3rd Order Differencing')
    plot_acf(df.diff().diff().diff().dropna(), ax=axes[3, 1], auto_ylims=True )
    adf = adfuller(df.diff().diff().diff().dropna()['Close'])
    axes[3, 1].text(0.1, 0.1, 'ADF p-value:'+str(round(adf[1],4)), {'fontsize': 20}, fontproperties = 'monospace')

    # 4th Differencing
    axes[4, 0].plot(df.diff().diff().diff().diff()); axes[4, 0].set_title('4th Order Differencing')
    plot_acf(df.diff().diff().diff().diff().dropna(), ax=axes[4, 1], auto_ylims=True )
    adf = adfuller(df.diff().diff().diff().diff().dropna()['Close'])
    axes[4, 1].text(0.1, 0.1, 'ADF p-value:'+str(round(adf[1],4)), {'fontsize': 20}, fontproperties = 'monospace')

    # 5th Differencing
    axes[5, 0].plot(df.diff().diff().diff().diff().diff()); axes[5, 0].set_title('5th Order Differencing')
    plot_acf(df.diff().diff().diff().diff().diff().dropna(), ax=axes[5, 1], auto_ylims=True )
    adf = adfuller(df.diff().diff().diff().diff().diff().dropna()['Close'])
    axes[5, 1].text(0.1, 0.1, 'ADF p-value:'+str(round(adf[1],4)), {'fontsize': 20}, fontproperties = 'monospace')
    

    plt.savefig('graphs/ARIMA_diff.png', dpi=300, bbox_inches = 'tight')


def find_order_p(df, d):
    df = df.drop(columns=['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'])
    
    fig, ax = plt.figure(figsize=(8,4)), plt.subplot(111)
    for _ in range(d):
        df = df.diff().dropna()

    plot_pacf(df, ax=ax, lags=5, auto_ylims=True); ax.set_title(f'{d}rd Order Differencing PACF')
    ax.set_xticks([0,1,2,3,4,5])
    plt.tight_layout()

    plt.savefig('graphs/ARIMA_p.png', dpi=300, bbox_inches = 'tight')



def find_order_q(df, d):
    df = df.drop(columns=['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'])
    
    fig, ax = plt.figure(figsize=(8,4)), plt.subplot(111)
    for _ in range(d):
        df = df.diff().dropna()

    plot_acf(df, ax=ax, lags=5, auto_ylims=True); ax.set_title(f'{d}rd Order Differencing ACF')
    ax.set_xticks([0,1,2,3,4,5])
    plt.tight_layout()

    plt.savefig('graphs/ARIMA_q.png', dpi=300, bbox_inches = 'tight')