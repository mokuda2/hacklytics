import streamlit as st

def contact_form():
    st.subheader("Enter details below to contact us")

    with st.form("form1", clear_on_submit=True):
        name = st.text_input("Full Name:")
        email = st.text_input("Email:")
        message = st.text_area("Message")
        submit = st.form_submit_button("Submit")