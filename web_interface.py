import streamlit as st
from arima_file import *
from PIL import Image

st.title("Forecasting Stock Prices")

### Plotting the stock price

stock = st.selectbox(
    'Select among popular stocks...',
    ('Apple', 'Tesla', 'Amazon', 'Google', 'Microsoft'))

stock_tick = st.text_input('... or write the ticker')


startDate= st.date_input("Enter the start date")
endDate= st.date_input("Enter the end date")

interval = st.selectbox(
    'Frequency of trades (for a frequence lower than 1h, restricted to data from the last 30 days and max. a week period))',
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

new_title = '<p style="font-family:sans-serif; color:blue; font-size: 30px;">Forecasting with ARIMA model</p>'
st.markdown(new_title, unsafe_allow_html=True)

new_title = '<p style="font-family:sans-serif; color:cornflowerblue; font-size: 22px;">First method: ML approach</p>'
st.markdown(new_title, unsafe_allow_html=True)
st.write('We look for the best ARIMA fit by splitting the data into a training (70%) and a test set (30%). We then perform a grid search to find the best order (p, d, q) for the ARIMA model which minimizes the MSE on the test set. Finally, we perform a rolling forecast (1 point forecasted). We re-create a new ARIMA model after each new observation of price stock.')

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
st.write('')
st.write('')


new_title = '<p style="font-family:sans-serif; color:cornflowerblue; font-size: 22px;">Second method: you do the fit!</p>'
st.markdown(new_title, unsafe_allow_html=True)

new_title = '<p style="font-family:sans-serif; color:cornflowerblue; font-size: 17px;">Step 1: Finding parameter d</p>'
st.markdown(new_title, unsafe_allow_html=True)

st.write('d corresponds to the number of times you have to differentiate the time series until it is stationary. We can use the Augmented Dickey-Fuller test to check if the time series is stationary. If the p-value is less than 0.05, we can reject the null hypothesis that the time series is not stationary.')
st.write('Choose the smallest d such that the time series is stationary.')

if st.button("Lets differentiate the time series!"):
    find_order_d(df)
    image = Image.open('graphs/ARIMA_diff.png')
    st.image(image)

d2 = st.text_input('Enter the value of d')

new_title = '<p style="font-family:sans-serif; color:cornflowerblue; font-size: 17px;">Step 2: Finding parameter p</p>'
st.markdown(new_title, unsafe_allow_html=True)

st.write(f'p corresponds to the number of p first lags that cross (above or beyond) the significance limit (blue surface) in the PACF plot of the d-order differentiated time series.')
if st.button("Show me the PACF!"):
    find_order_p(df, int(d2))
    image = Image.open('graphs/ARIMA_p.png')
    st.image(image)

st.write('If you have a doubt (lag close to the significance level), always go for the smallest p')
p2 = st.text_input('Enter the value of p')

new_title = '<p style="font-family:sans-serif; color:cornflowerblue; font-size: 17px;">Step 3: Finding parameter q</p>'
st.markdown(new_title, unsafe_allow_html=True)

st.write('Exactly the same as before but with the ACF plot')
if st.button("Show me the ACF!"):
    find_order_q(df, int(d2))
    image = Image.open('graphs/ARIMA_q.png')
    st.image(image)

q2 = st.text_input('Enter the value of q')

if st.button("We're all set! Fit & Forecast"):
    st.write('Fitting an ARIMA model... (~ 20 seconds)')
    model_predictions, test_data = fit_ARIMA(df, int(p2), int(d2), int(q2))
    rolling_forecast_ARIMA(ticker, df, model_predictions, test_data, between_tick=between_tick)
    image = Image.open('graphs/ARIMA.png')
    st.image(image)