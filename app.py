import streamlit as st
import pandas as pd
import pickle
import base64

# Load the model using caching for better performance
@st.cache_resource
def load_model():
    with open('xgb_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Optional: Remove if you're not using a local image file
@st.cache_data  # Cache data loading function
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Example usage of the local image, if you choose to use it
# img = get_img_as_base64("image.jpg")

# Set background with CSS (using an online image)
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

    # Input fields
    amount = st.text_input('Enter the Total Amount of Transaction')
    oldbalanceOrg = st.text_input('Enter The old balance on the origin account before the transaction')
    newbalanceOrig = st.text_input('Enter The new balance on the origin account after the transaction')
    oldbalanceDest = st.text_input('Enter The old balance on the destination account before the transaction')
    newbalanceDest = st.text_input('Enter The new balance on the destination account after the transaction')

    # Placeholder for prediction
    prediction = ''

    if st.button('Predict'):
        if not amount or not oldbalanceOrg or not newbalanceOrig or not oldbalanceDest or not newbalanceDest:
            st.error("Please fill in all the fields.")
        else:
            try:
                input_data = [selected_value, float(amount), float(oldbalanceOrg), float(newbalanceOrig), float(oldbalanceDest), float(newbalanceDest)]
                prediction = fraud_detection(input_data)
                st.success(prediction)


if __name__ == '__main__':
    main()
