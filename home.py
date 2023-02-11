import streamlit as st
from PIL import Image

def home_info():
    col1, col2, col3 = st.columns([1,6,1])
    image = Image.open('img/finance.png')
    with col1:
        st.write("")

    with col2:
        st.image(image)

    with col3:
        st.write("")
    st.write('FinCast is a financial forecasting platform that allows you to analyse historical market data. It uses Machine Learning to understand the stock market and Cryptocurrencies.')
    st.write('**DISCLAIMER**')
    st.write('*The Content of FinCast is for informational purposes only, you should not construe any such information or other material as legal, tax, investment, financial, or other advice.*')
