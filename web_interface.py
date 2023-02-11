import streamlit as st
from arima_file import *
from PIL import Image
from datetime import datetime

st.title("Welcome to FinCast's Forecasting Platform")

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