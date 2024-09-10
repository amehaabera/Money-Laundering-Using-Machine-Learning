import base64
import pickle
import streamlit as st
import pandas as pd

# Load the saved model
model = pickle.load(open('best_model.pkl', 'rb'))

@st.cache_data
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
    feature_names = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']

    # Create a DataFrame with a single row using input_data
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Debugging: print input DataFrame to check values
    st.write("Input DataFrame:", input_df)

    # Make a prediction
    prediction = model.predict(input_df)

    if prediction[0] == 0:
        return 'Not a fraudulent transaction'
    else:
        return 'Fraudulent transaction'

def main():
    st.title('**Fraudulent Transaction Detection System!!**')

    options = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    values = [2, 4, 1, 5, 3]

    # Create a selectbox for transaction type
    transaction_type = st.selectbox('Select the Type of Transaction', options)
    
    # Map the selected transaction type to its corresponding value
    type_to_value = dict(zip(options, values))
    selected_value = type_to_value[transaction_type]

    # Input fields
    step = st.text_input('Enter the Step')
    amount = st.text_input('Enter the Total Amount of Transaction')
    oldbalanceOrg = st.text_input('Enter The old balance on the origin account before the transaction')
    newbalanceOrig = st.text_input('Enter The new balance on the origin account after the transaction')
    oldbalanceDest = st.text_input('Enter The old balance on the destination account before the transaction')
    newbalanceDest = st.text_input('Enter The new balance on the destination account after the transaction')
    isFlaggedFraud = st.text_input('Is Flagged Fraud')

    prediction = ''

    if st.button('Predict'):
        try:
            # Convert inputs to floats and create input data
            input_data = [
                float(step),
                selected_value,
                float(amount),
                float(oldbalanceOrg),
                float(newbalanceOrig),
                float(oldbalanceDest),
                float(newbalanceDest),
                float(isFlaggedFraud)
            ]
            prediction = fraud_detection(input_data)
            st.success(prediction)
        except ValueError as e:
            st.error(f"Error in input values: {e}")

if __name__ == '__main__':
    main()
