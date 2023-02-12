import matplotlib.pyplot as plt
import xgboost as xgb

def model_XGBOOST(tick):
    if tick == "AAPL":
        model = xgb.Booster()
        model.load_model('Models/gboost_saved_model/AAPL_xgboost_model.bin')
    if tick =="TSLA":
        model = xgb.Booster()
        model.load_model('Models/gboost_saved_model/TSLA_xgboost_model.bin')
    if tick == "AMZN":
        model = xgb.Booster()
        model.load_model('Models/gboost_saved_model/AMZN_xgboost_model.bin')
    if tick == "GOOGL":
        model = xgb.Booster()
        model.load_model('Models/gboost_saved_model/GOOG_xgboost_model.bin')
    if tick == "MSFT":
        model = xgb.Booster()
        model.load_model('Models/gboost_saved_model/MSFT_xgboost_model.bin')
    if tick == "BTC-USD":
        model = xgb.Booster()
        model.load_model('Models/gboost_saved_model/BTC-USD_xgboost_model.bin')
    if tick == "ETH-USD":
        model = xgb.Booster()
        model.load_model('Models/gboost_saved_model/ETH-USD_xgboost_model.bin')
    if tick == "DOGE_USD":
        model = xgb.Booster()
        model.load_model('Models/gboost_saved_model/DOGE-USD_xgboost_model.bin')
    if tick == "XRP_USD":
        model = xgb.Booster()
        model.load_model('Models/gboost_saved_model/XRP-USD_xgboost_model.bin')
    return model 

def feature_engineering(df): 
  # The model will not accept datetime, hence create a feature for each date part
  df.reset_index(drop=False, inplace=True)
  print(df.head())
  print(df["Date"])
 # df["Date"] = df["Date"].apply(lambda x: datetime.fromtimestamp( (x - 25569) *86400.0))
  df["Year"] = df["Date"].dt.year
  df["Month"] = df["Date"].dt.month
  df["Day"] = df["Date"].dt.day
  # df.drop(["Close"], inplace=True, axis=1)
  print(df.head())
  return df

def split_data(df):
#   train_split = 0.7
#   # Set the date at which to split train and eval data
#   # Of the unique dates available, pick the split between train and eval dates
#   dates_avail = df["Date"].unique()
#   split_date_index = int(dates_avail.shape[0] * train_split)
#   split_date = dates_avail[split_date_index]
#   # Train data is on or before the split date
#   train_df = df.query("Date <= @split_date")
#   # And eval data is after
#   eval_df = df.query("Date > @split_date")
  train_df, eval_df = df[0:int(len(df)*0.7)], df[int(len(df)*0.7):]
  features = ["Year", "Month", "Day","Open", "High", "Low", "Close", "Adj Close", "Volume", ]
  label = ["Adj Close"]
  x_train = train_df[features]
  y_train = train_df[label]
  x_eval = eval_df[features]
  y_eval = eval_df[label]
  return x_train, y_train, x_eval, y_eval 

def fit_XGBOOST(df, model):
    df = feature_engineering(df)
    _, _, x_eval,_ = split_data(df)
    # convert to xgb.DMatrix
    x_eval_DM= xgb.DMatrix(x_eval)
    predicted_prices = model.predict(x_eval_DM)
    actual_prices = x_eval['Close'].values
    return predicted_prices, actual_prices


def rolling_forecast_XGBOOST(ticker,df, predicted_prices, actual_prices, between_tick=20):
    test_set_range = df[int(len(df)*0.7):].index
    plt.figure(figsize=(20,10))
    plt.rc('axes', axisbelow=True)
    plt.plot(test_set_range, actual_prices, color='red', label=f"Actual {ticker} price")
    plt.plot(test_set_range, predicted_prices, color= 'blue', marker='o', linestyle='dashed',label=f"Predicted {ticker} Price")
    plt.title(f"{ticker} Price Prediction")
    plt.xlabel("Date")
    plt.ylabel(f"Prices")
    plt.xticks(test_set_range[::between_tick], df.Date[0:len(actual_prices):between_tick], rotation=45)
    plt.legend()
    plt.savefig(f'graphs/XGBOOST.png', dpi=300, bbox_inches = 'tight')
