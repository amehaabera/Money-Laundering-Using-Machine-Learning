import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import joblib
import pickle
import warnings
import altair as alt
import sklearn
import base64
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')

model_path = pickle.load(open('xgb_model.pkl', 'rb'))
# Load the model using joblib

@st.cache_data  # Updated caching method for data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()



img = get_img_as_base64("image.jpg")


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
    background-size: 280%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}
</style>
"""


st.markdown(page_bg_img, unsafe_allow_html=True)



# define function to preprocess user input
def preprocess_input(payment, amount, oldbalanceOrg, newbalanceOrg, oldbalanceDest, newbalanceDest):
    # create dataframe with user input
    input_df = pd.DataFrame({'type': [payment], 
                             'amount': [amount], 
                             'oldbalanceOrg': [oldbalanceOrg], 
                             'newbalanceOrg': [newbalanceOrg], 
                             'oldbalanceDest': [oldbalanceDest], 
                             'newbalanceDest': [newbalanceDest]})
  
    # return preprocessed input
    return input_df

# define Streamlit app
def app():
    # set app title
    st.title('Money Laundering Detector')
    
    # add sidebar to select transaction type
    types = st.sidebar.subheader("""
                 Enter Type of Transfer Made:\n\n\n\n
                 0 for 'CASH_IN' Transaction\n 
                 1 for 'CaASH_OUT' Transaction\n 
                 2 for 'DEBIT' Transaction\n
                 3 for 'PAYMENT' Transaction\n  
                 4 for 'TRANSFER' Transaction\n""")
    types = st.sidebar.selectbox("",(0,1,2,3,4))
    x = ''
    if types == 0:
        x = 'CASH_IN'
    if types == 1:
        x = 'CASH_OUT'
    if types == 2:
        x = 'DEBIT'
    if types == 3:
        x = 'PAYMENT'
    if types == 4:
        x =  'TRANSFER'

    # define input fields
    payment = x
    amount = st.number_input('Amount')
    oldbalanceOrg = st.number_input('Old Balance (Origin)')
    newbalanceOrg = st.number_input('New Balance (Origin)')
    oldbalanceDest = st.number_input('Old Balance (Destination)')
    newbalanceDest = st.number_input('New Balance (Destination)')

    # preprocess user input
    input_data = preprocess_input(payment, amount, oldbalanceOrg, newbalanceOrg, oldbalanceDest, newbalanceDest)
    
    # make prediction
    prediction = model.predict(input_data)
    
    # display result
    if prediction[0] == 0:
        st.write('The Person is Fraud')
    else:
        st.write('The Person Not Fraud')

    # display input data
    st.write('Input Data:')
    st.write(input_data)

if __name__ == '__main__':
    app()
