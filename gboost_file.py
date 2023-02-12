import tensorflow as tf

def model_XGBOOST(tick):
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

