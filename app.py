import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the saved model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Check the type of the loaded model
st.write(f"Loaded model type: {type(model)}")  # This will help you debug if the model is loaded correctly

# Define feature names (make sure this matches the order used during training)
feature_names = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']

# Preprocess input data to match the feature transformations applied during training
def preprocess_data(data):
    input_df = data.copy()
    preprocessed_data = pd.DataFrame(columns=feature_names)
    
    # Label encode 'type' column
    le = LabelEncoder()
    le.fit(['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])  # Make sure this matches the training set
    
    preprocessed_data['step'] = input_df['step']
    preprocessed_data['type'] = le.transform(input_df['type'])  # Transform categorical 'type'
    preprocessed_data['amount'] = np.log1p(input_df['amount'])  # Log-transform 'amount'
    preprocessed_data['oldbalanceOrg'] = input_df['oldbalanceOrg']
    preprocessed_data['newbalanceOrig'] = input_df['newbalanceOrig']
    preprocessed_data['oldbalanceDest'] = input_df['oldbalanceDest']
    preprocessed_data['newbalanceDest'] = input_df['newbalanceDest']
    preprocessed_data['isFlaggedFraud'] = input_df['isFlaggedFraud']
    
    return preprocessed_data

# Make predictions using the loaded model
def predict(data):
    preprocessed_data = preprocess_data(data)
    predictions = model.predict(preprocessed_data)
    return predictions

# Create a multi-page layout using the sidebar
def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home", "App", "About"])

    if selection == "Home":
        st.title("Anti-money laundering (AML) transaction monitoring software System")
        st.write("""
        This application uses machine learning models to predict the likelihood of a transaction being fraudulent.
        You can enter the transaction details and the system will predict if the transaction is **Normal** or **Fraudulent**.
        """)
        st.write("""
        ### Features:
        - Enter details of transactions such as payment type, amount, and balances.
        - Get real-time predictions about the likelihood of fraudulent transactions.
        """)

    elif selection == "App":
        st.title("**Money Laundering Transaction Detection System**")

        # Input form
        st.header('Enter transaction details:')
        _type = st.selectbox(label='Select payment type', options=['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])
        step = st.number_input('Step', min_value=1)
        amount = st.number_input('Amount', min_value=0.0)
        oldbalanceOrg = st.number_input('Old Balance Origin')
        newbalanceOrig = st.number_input('New Balance Origin', value=oldbalanceOrg)
        oldbalanceDest = st.number_input('Old Balance Destination')
        newbalanceDest = st.number_input('New Balance Destination', value=oldbalanceDest)
        isFlaggedFraud = st.number_input('Is Flagged Fraud (0 or 1)', min_value=0, max_value=1)

        # Create DataFrame for input
        input_data = pd.DataFrame({
            'step': [step],
            'type': [_type],
            'amount': [amount],
            'oldbalanceOrg': [oldbalanceOrg],
            'newbalanceOrig': [newbalanceOrig],
            'oldbalanceDest': [oldbalanceDest],
            'newbalanceDest': [newbalanceDest],
            'isFlaggedFraud': [isFlaggedFraud]
        }, columns=feature_names)

        # Predict button
        if st.button('Predict'):
            try:
                predictions = predict(input_data)
                is_fraud = 'Fraud' if predictions[0] == 1 else 'Normal'
                
                # Display the result in green if 'Normal', red if 'Fraud'
                if is_fraud == 'Normal':
                    st.markdown(f"<h3 style='color:green; font-weight:bold;'>Predicted Class: {is_fraud}</h3>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 style='color:red; font-weight:bold;'>Predicted Class: {is_fraud}</h3>", unsafe_allow_html=True)

            except ValueError:
                st.error("Please enter valid numeric values for all fields.")
            except Exception as e:
                st.error(f"Error in prediction: {e}")

    elif selection == "About":
        st.title("About the Application")
        st.write("""
        This application was developed to help detect potential money laundering transactions using machine learning.
        The model is trained on a dataset of historical transactions and uses various transaction features to make predictions.
        """)
        st.write("""
        ### Developer Contact:
        - **Developer**: Ameha Abera Kidane
        - **Email**: amehaabera@gmail.com
        - **Linkedin**: https://www.linkedin.com/in/ameha-abera-kidane/
        - **GitHub**: [Your GitHub Profile](https://github.com/amehaabera)
        """)

if __name__ == '__main__':
    main()
