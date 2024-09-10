import base64
import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the saved model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Check the type of the loaded model
st.write(f"Loaded model type: {type(model)}")  # This will help you debug

# Define feature names
feature_names = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']

# Preprocess input data
def preprocess_data(data):
    input_df = data.copy()
    preprocessed_data = pd.DataFrame(columns=feature_names)
    
    # Initialize LabelEncoder for 'type'
    le = LabelEncoder()
    # Fit LabelEncoder with the known transaction types
    le.fit(['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])  # Use the actual types your model was trained with
    
    preprocessed_data['step'] = input_df['step']
    preprocessed_data['type'] = le.transform(input_df['type'])
    preprocessed_data['amount'] = np.log1p(input_df['amount'])
    preprocessed_data['oldbalanceOrg'] = input_df['oldbalanceOrg']
    preprocessed_data['newbalanceOrig'] = input_df['newbalanceOrig']
    preprocessed_data['oldbalanceDest'] = input_df['oldbalanceDest']
    preprocessed_data['newbalanceDest'] = input_df['newbalanceDest']
    preprocessed_data['isFlaggedFraud'] = input_df['isFlaggedFraud']  # Include this feature
    
    return preprocessed_data

# Make predictions using the loaded model
def predict(data):
    preprocessed_data = preprocess_data(data)
    predictions = model.predict(preprocessed_data)
    return predictions

# Create the Streamlit app
def main():
    st.title('Fraud Detection App')

    # Create input form
    st.header('Enter transaction details:')
    _type = st.selectbox(label='Select payment type', options=['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])
    step = st.number_input('Step', min_value=1)
    amount = st.number_input('Amount', min_value=0.0)
    oldbalanceOrg = st.number_input('Old Balance Origin')
    newbalanceOrig = st.number_input('New Balance Origin', value=oldbalanceOrg)  # Allow user to enter this separately
    oldbalanceDest = st.number_input('Old Balance Destination')
    newbalanceDest = st.number_input('New Balance Destination', value=oldbalanceDest)  # Allow user to enter this separately
    isFlaggedFraud = st.number_input('Is Flagged Fraud (0 or 1)', min_value=0, max_value=1)

    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'type': [_type],
        'step': [step],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest],
        'isFlaggedFraud': [isFlaggedFraud]
    }, columns=feature_names)

    # Make predictions on button click
    if st.button('Predict'):
        try:
            predictions = predict(input_data)
            is_fraud = 'Fraud' if predictions[0] == 1 else 'Normal'
            st.write(f'Predicted Class: {is_fraud}')
        except Exception as e:
            st.error(f"Error in prediction: {e}")

# Run the app
if __name__ == '__main__':
    main()
