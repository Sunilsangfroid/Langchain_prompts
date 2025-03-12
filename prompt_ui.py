from langchain import OpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.header('Research Tool')

user_input = st.text_input('Enter your Prompt:')

if st.button('Summarize'):
    result = model.invoke(user_input)
    st.write('Some random text')
