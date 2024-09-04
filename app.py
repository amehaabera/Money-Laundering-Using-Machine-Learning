import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import joblib
import warnings
import altair as alt
import sklearn
import base64
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')

# Load the trained model from the file
model_path = r"/workspaces/Money-Laundering-Using-Machine-Learning/xgb_model.pkl"
model = joblib.load(model_path)  
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



def fraud_detection(input_data):
    # Define feature names
    feature_names = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

    # Convert input_data to a dictionary with feature names
    input_dict = dict(zip(feature_names, input_data))

    # Create a DataFrame with a single row using input_dict
    input_df = pd.DataFrame(input_dict, index=[0])

    # Make a prediction
    prediction = model.predict(input_df)

    if prediction[0] == 0:
        return 'Not a fraudulent transaction'
    else:
        return 'Fraudulent transaction'

def main():
    st.title('**Money Laundering Transaction Detection System!!**')

    options = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    values = [2, 4, 1, 5, 3]

    # Create a selectbox for transaction type
    type = st.selectbox('Select the Type of Transaction', options)

    # Use a dictionary to map the selected transaction type to its corresponding value
    type_to_value = dict(zip(options, values))
    selected_value = type_to_value[type]

    amount = st.text_input('Enter the Total Amount of Transaction')
    oldbalanceOrg = st.text_input('Enter The old balance on the origin account before the transaction')
    newbalanceOrig = st.text_input('Enter The new balance on the origin account after the transaction')
    oldbalanceDest = st.text_input('Enter The old balance on the destination account before the transaction')
    newbalanceDest = st.text_input('Enter The new balance on the destination account after the transaction')

    prediction = ''

    if st.button('Predict'):
        input_data = [selected_value, float(amount), float(oldbalanceOrg), float(newbalanceOrig), float(oldbalanceDest), float(newbalanceDest)]
        prediction = fraud_detection(input_data)
        st.success(prediction)

if __name__ == '__main__':
    main()
