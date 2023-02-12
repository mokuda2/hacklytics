import streamlit as st
from arima_file import *
from lstm_file import *
from contact_us import *
from gboost_file import *
from PIL import Image
from datetime import datetime
from home import *
from streamlit_option_menu import option_menu
from datetime import timedelta
import streamlit.components.v1 as html
import base64

with st.sidebar:
    choose = option_menu("FinCast", ["Home","ARIMA Forecast", "LSTM Forecast", "XGBoost Forecast", "Order Book Forecast",  "Contact"],
                         icons=['house', 'bar-chart-line-fill', 'bar-chart-line','bar-chart-steps', 'bar-chart-line-fill','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#41b3a3"},
    }
    )

if choose == "Home": 
    st.title("Welcome to FinCast's Forecasting Platform")
    home_info()

    st.write('')

if choose == "ARIMA Forecast":

    st.title("Welcome to FinCast's ARIMA Model")

    st.write('')


    new_title = '<p style="font-family:cursive; color:blue; font-size: 30px;">First, lets select a dataset.</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    ### Plotting the stock price

    stock = st.selectbox(
        'Select among popular stocks...',
        ('Apple', 'Tesla', 'Amazon', 'Google', 'Microsoft'))

    stock_tick = st.text_input('... or write the ticker')


    startDate= st.date_input("Enter the start date")

    endDate= st.date_input("Enter the end date (no future dates allowed here)")

    startDate = datetime.combine(startDate, datetime.min.time())
    endDate = datetime.combine(endDate, datetime.min.time())

    if endDate> datetime.now():
        new_title = '<p style="font-family:cursive; color:red; font-size: 15px;">Please enter a valid end date.</p>'
        st.markdown(new_title, unsafe_allow_html=True)


    interval = st.selectbox(
        'Frequency of trades (for a frequency lower than 1h, restricted to data from the last 30 days and max. a week period). \n Please make sure you have at least 30 data points in total (otherwise, it is likely an error occurs).',
        ('1d', '1w', '1mo', '1m', '30m', '1h'))


    between_tick = st.slider('Time period between two date ticks', min_value=1, max_value=90, value=10, step=1)

    tickers = {'Apple': 'AAPL', 'Tesla': 'TSLA', 'Amazon': 'AMZN', 'Google': 'GOOGL', 'Microsoft': 'MSFT'}

    if len(stock_tick) !=0:
        ticker = stock_tick.upper()
    else:
        ticker = tickers[stock]

    df = get_data(ticker, startDate, endDate, interval)

    st.write('Make sure you click on the button below to make predictions later on!')
    if st.button("See the stock price!"):
        plot_stock(df, ticker, between_tick=between_tick)
        image = Image.open('graphs/stock.png')
        st.image(image)

    #add blank space
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')


    ### Fitting an ARIMA model

    new_title = '<p style="font-family: cursive; color:blue; font-size: 30px;">Predicting past data with ARIMA model</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    st.write('')

    new_title = '<p style="font-family: cursive; color:cornflowerblue; font-size: 22px;">First method: ML approach</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.write('We look for the best ARIMA fit by splitting the data you selected above into a training (70%) and a test set (30%). We then perform a grid search to find the best order (p, d, q) for the ARIMA model which minimizes the MSE on the test set. Finally, we perform a 1-point rolling forecast to see how well this model can fit the data (a new ARIMA model is created after each new observation of price stock)')

    if st.button("Fit an ARIMA!"):
        st.write('Fitting an ARIMA model... (~ 20 seconds)')
        p, d, q = model_ARIMA(df)
        st.write("Cheers, ARIMA fitted!")
        st.write(f"The best order is: (p,d,q)={p}, {d}, {q}. Now, lets forecast (it's running...)")
        model_predictions, test_data = fit_ARIMA(df, p, d, q)
        rolling_forecast_ARIMA(ticker, df, model_predictions, test_data, between_tick=between_tick)
        image = Image.open('graphs/ARIMA.png')
        st.image(image)



    st.write('')
    st.write('')


    new_title = '<p style="font-family: cursive; color:cornflowerblue; font-size: 22px;">Second method: you do the fit!</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    new_title = '<p style="font-family: cursive; color:cornflowerblue; font-size: 17px;">Step 1: Finding parameter d</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    st.write('d corresponds to the number of times you have to differentiate the time series until it is stationary. We can use the Augmented Dickey-Fuller test to check if the time series is stationary. If the p-value is less than 0.05, we can reject the null hypothesis that the time series is not stationary.')
    st.write('Choose the smallest d such that the time series is stationary.')

    if st.button("Lets differentiate the time series!"):
        find_order_d(df)
        image = Image.open('graphs/ARIMA_diff.png')
        st.image(image)

    d2 = st.text_input('I found d=')

    st.write('')
    st.write('')


    new_title = '<p style="font-family: cursive; color:cornflowerblue; font-size: 17px;">Step 2: Finding parameter p</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    st.write(f'p corresponds to the number of (p-1) vertical bars that cross (above or beyond) the significance limit (blue surface) in the PACF plot of the d-order differentiated time series.')
    st.write('Example: if only the left-most vertical bar (labeled by 0) crosses the significance level, then p=0.')

    if st.button("Show me the PACF!"):
        find_order_p(df, int(d2))
        image = Image.open('graphs/ARIMA_p.png')
        st.image(image)

    st.write('If you have a doubt (bar height close to the significance level), always go for the smallest p.')
    p2 = st.text_input('I found p=')


    st.write('')
    st.write('')

    new_title = '<p style="font-family: cursive; color:cornflowerblue; font-size: 17px;">Step 3: Finding parameter q</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    st.write('Exactly the same as before but with the ACF plot.')
    if st.button("Show me the ACF!"):
        find_order_q(df, int(d2))
        image = Image.open('graphs/ARIMA_q.png')
        st.image(image)

    q2 = st.text_input('I found q=')
    if q2 !='':
        st.write("Note that the set of parameters might be different from the ones found with the ML approach. The ML approach is more likely to overfit the data. The choice is up to you!")

    if st.button("We're all set! Fit & Forecast"):
        st.write('Fitting an ARIMA model... (~ 20 seconds)')
        model_predictions, test_data = fit_ARIMA(df, int(p2), int(d2), int(q2))
        rolling_forecast_ARIMA(ticker, df, model_predictions, test_data, between_tick=between_tick)
        image = Image.open('graphs/ARIMA.png')
        st.image(image)


    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    new_title = '<p style="font-family: cursive; color:blue; font-size: 30px;">Forecasting Strategy</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    new_title = '<p style="font-family: cursive; color:cornflowerblue; font-size: 17px;">How to use the ARIMA model?</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    st.write('The ARIMA model is a forecasting model. It is not a trading strategy. It is up to you to decide how to use it. Here are a few ideas:')
    st.write('Use the ARIMA model to forecast the price of the stock one step ahead of your time period (max. 2, otherwise the model is not reliable). If the forecasted price is higher than the current price, buy the stock. If the forecasted price is lower than the current price, sell the stock.')
    st.write('This can be useful if you want to forecast the tendency of the market at its most liquid hours of the day.')
    st.write('For instance, the last two hours of the session of the U.S equity market, between 2pm and 4pm, are the most liquid of all. This period is the best time to buy or sell a large quantity of assets, because market conditions are better (lower transaction costs, lower volatility, ...). This is why estimating in advance the behavior of an asset over this period allows to optimize the whole portfolio.')

if choose == "LSTM Forecast": 
    st.title("Welcome to FinCast's LSTM Model")

    st.write('')

    new_title = '<p style="font-family:cursive; color:blue; font-size: 30px;">First, lets select a dataset.</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    ### Plotting the stock price

    stock = st.selectbox(
        'Select among popular stocks...',
        ('Apple', 'Tesla', 'Amazon', 'Google', 'Microsoft', 'Bitcoin', 'Etherium', 'Dogecoin', 'Ripple'))

    stock_tick = st.text_input('... or write the ticker')


    startDate= st.date_input("Enter the start date")

    endDate= st.date_input("Enter the end date (no future dates allowed here)")

    startDate = datetime.combine(startDate, datetime.min.time())
    endDate = datetime.combine(endDate, datetime.min.time())

    if endDate> datetime.now():
        new_title = '<p style="font-family:cursive; color:red; font-size: 15px;">Please enter a valid end date.</p>'
        st.markdown(new_title, unsafe_allow_html=True)


    interval = st.selectbox(
        'Frequency of trades (for a frequency lower than 1h, restricted to data from the last 30 days and max. a week period). \n Please make sure you have at least 30 data points in total (otherwise, it is likely an error occurs).',
        ('1d', '1w', '1mo', '1m', '30m', '1h'))


    between_tick = st.slider('Time period between two date ticks', min_value=1, max_value=90, value=10, step=1)

    tickers = {'Apple': 'AAPL', 'Tesla': 'TSLA', 'Amazon': 'AMZN', 'Google': 'GOOGL', 'Microsoft': 'MSFT', 'Bitcoin': 'BTC-USD', 'Etherium': 'ETH-USD', 'Dogecoin': 'DOGE-USD', 'Ripple':'XRP'}

    if len(stock_tick) !=0:
        ticker = stock_tick.upper()
    else:
        ticker = tickers[stock]

    df = get_data(ticker, startDate, endDate, interval)

    st.write('Make sure you click on the button below to make predictions later on!')
    if st.button("See the stock price!"):
        plot_stock(df, ticker, between_tick=between_tick)
        image = Image.open('graphs/stock.png')
        st.image(image)

    #add blank space
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')


    ### Fitting an LSTM model

    new_title = '<p style="font-family: cursive; color:blue; font-size: 30px;">Predicting past data with LSTM model</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    st.write('')

    new_title = '<p style="font-family: cursive; color:cornflowerblue; font-size: 22px;"> STEP 1: Long-Short Term Memory Network approach</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.write('LSTM stands for Long Short-Term Memory Network. LSTM is a machine learning model often used in sequence prediction problems.\nIt is a type of Recurrent Neural Network.')
    st.write('In this case we look for the best LSTM fit by splitting the data you selected above into a training (70%) and a test set (30%).')

    if st.button("Fit an LSTM!"):
        st.write('Fitting an LSTM model... (max 20 seconds)')
        model = model_LSTM(ticker)
        st.write("Cheers, LSTM fitted!")
        predicted_prices, actual_prices, next_day_pred = fit_LSTM(df, model)
        rolling_forecast_LSTM(ticker, df, predicted_prices, actual_prices, between_tick=between_tick)
        image = Image.open('graphs/LSTM.png')
        st.image(image)
        next_day = endDate  + timedelta(days=1)
        new_title = '<p style="font-family: cursive; color:cornflowerblue; font-size: 22px;"> STEP 2: Predicting next day</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.write(f'Predicting next day ({next_day}) {ticker} price: {next_day_pred[0][0]} USD')


if choose == "XGBoost Forecast":
    st.title("Welcome to FinCast's Extreme Gradient Boosting Model")
    st.write('')

    new_title = '<p style="font-family:cursive; color:blue; font-size: 30px;">First, lets select a dataset.</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    ### Plotting the stock price

    stock = st.selectbox(
        'Select among popular stocks...',
        ('Apple', 'Tesla', 'Amazon', 'Google', 'Microsoft', 'Bitcoin', 'Etherium', 'Dogecoin', 'Ripple'))

    stock_tick = st.text_input('... or write the ticker')


    startDate= st.date_input("Enter the start date")

    endDate= st.date_input("Enter the end date (no future dates allowed here)")

    startDate = datetime.combine(startDate, datetime.min.time())
    endDate = datetime.combine(endDate, datetime.min.time())

    if endDate> datetime.now():
        new_title = '<p style="font-family:cursive; color:red; font-size: 15px;">Please enter a valid end date.</p>'
        st.markdown(new_title, unsafe_allow_html=True)


    interval = st.selectbox(
        'Frequency of trades (for a frequency lower than 1h, restricted to data from the last 30 days and max. a week period). \n Please make sure you have at least 30 data points in total (otherwise, it is likely an error occurs).',
        ('1d', '1w', '1mo', '1m', '30m', '1h'))


    between_tick = st.slider('Time period between two date ticks', min_value=1, max_value=90, value=10, step=1)

    tickers = {'Apple': 'AAPL', 'Tesla': 'TSLA', 'Amazon': 'AMZN', 'Google': 'GOOGL', 'Microsoft': 'MSFT', 'Bitcoin': 'BTC-USD', 'Etherium': 'ETH-USD', 'Dogecoin': 'DOGE-USD', 'Ripple':'XRP'}

    if len(stock_tick) !=0:
        ticker = stock_tick.upper()
    else:
        ticker = tickers[stock]

    df = get_data(ticker, startDate, endDate, interval)

    st.write('Make sure you click on the button below to make predictions later on!')
    if st.button("See the stock price!"):
        plot_stock(df, ticker, between_tick=between_tick)
        image = Image.open('graphs/stock.png')
        st.image(image)

    #add blank space
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')


    ### Fitting an XGBoost model
    new_title = '<p style="font-family: cursive; color:blue; font-size: 30px;">Predicting past data with XGBoost model</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    st.write('')

    new_title = '<p style="font-family: cursive; color:cornflowerblue; font-size: 22px;"> Extreme Gradient Boostinng approach</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.write('XGBoost stands for Extreme Gradient Boosting. XGBoost is an algorithm that has been recently applied to machine learning and it is often used for both regressions and clssification problems.')
    st.write('In this case we look for the best LSTM fit by splitting the data you selected above into a training (70%) and a test set (30%).')

    if st.button("Fit an XGBoost!"):
        st.write('Fitting an XGBoost model... (max 20 seconds)')
        model = model_XGBOOST(ticker)
        st.write("Cheers, XGboost fitted!")
        predicted_prices, actual_prices = fit_XGBOOST(df, model)
        rolling_forecast_XGBOOST(ticker, df, predicted_prices, actual_prices, between_tick=between_tick)
        image = Image.open('graphs/XGBOOST.png')
        st.image(image)

if choose == "Order Book Forecast":
    st.title("Welcome to FinCast's Order Book Forecast Section")

    file_ = open("graphs/Video_230212082114.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<div style="display: flex; justify-content: center;"><img style="width:700px; height:auto;" src="data:image/gif;base64,{data_url}" alt="cat gif"></div>',
        unsafe_allow_html=True,
    )

    st.write("Here's an example of our bid-ask spread animation for the Amazon stock.")

    st.write("In this section we focus on forecasting mid-quote prices and time arrival of orders.")






if choose == "Contact":
    st.title("FinCast's Contact Form")
    contact_form()
    st.write('')