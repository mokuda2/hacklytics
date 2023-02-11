# FinCast

FinCast enables financial forecasting of stock prices and cryptocurrencies. For this project, we have used stock market data and cryptocurrency historical data from Yahoo Finance.


# Requirement

We suggest to set up a virtual environment in the hacklytics folder by doing the following:

```
virtualenv --python python3 venv
```

To activate it, simply run

```
source venv/bin/activate
```

To run the application, you will need: 

* `streamlit` that can be installed using `pip install streamlit`
* `pandas_datareader` that can be installed using `pip install pandas_datareader`

* `tensorflow` that can be installed using `pip install tensorflow`
* `statsmodels'` that can be installed using `pip install statsmodels`
* `sklearn` that can be installed using `pip install scikit-learn`
* `matplotlib` that can be installed using `pip install matplotlib`
* `yfinance` that can be installed using `pip install yfinance`
* `streamlit-option-menu` that can be isntalled using `pip install streamlit-option-menu` 

When you have installed all the required libraries, simply run 

```
streamlink run web_interface.py
```

# Data

To load the data, we have the function: 

```
import yfinance as yf
# getting latest data
def get_data(stocks, start, end):
    df = pdr.get_data_yahoo(stocks, start, end)
    return df
```

# LSTM Network Model

The machine learning model to predict stock prices can be found in the notebook below: 

```
./Notebook/FinCast.ipynb
```

The machine learning model to predict cryptocurrency prices can be found in the notebook below: 

```
./Notebook/CryptoFinCast.ipynb
```

We have used long short-term memory networks to work on this project.
