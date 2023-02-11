# FinCast

FinCast enable financial forecasting of stock prices and cryptocurrencies. For this project we have used real-time Stock Market Data and Cryptocurrency historical data from yahoo Finance .

# Data

To load the data we have the function: 

```
import yfinance as yf
# getting latest data
def get_data(stocks, start, end):
    df = pdr.get_data_yahoo(stocks, start, end)
    return df
```

# LSTM Network Model

The machine learning model code can be found in the notebook in 

```
./Notebook/FinCast.ipynb
```
