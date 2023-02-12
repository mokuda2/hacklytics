import streamlit as st
from PIL import Image

def contact_form():
    st.subheader("Enter details below to contact us")
    st.write("")
    st.write("")
    st.write("")
    image = Image.open('img/finance.png')
    col1, col2, col3 = st.columns([1,6,1])
    with col1:
        st.write("")

    with col2:
        st.image(image)

    with col3:
        st.write("")
    st.write("")
    st.write("")
    st.write("")
    with st.form("form1", clear_on_submit=True):
        name = st.text_input("Full Name:")
        email = st.text_input("Email:")
        message = st.text_area("Message")
        submit = st.form_submit_button("Submit")
    
    






