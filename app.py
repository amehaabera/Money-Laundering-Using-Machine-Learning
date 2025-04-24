import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the saved model
with open('best_model4.pkl', 'rb') as file:
    model = pickle.load(file)

# Define feature names in EXACT order expected by the model
FEATURE_NAMES = [
    'Old Balance Org', 'New Balance Orig', 'Amount in Birr', 'Account Type',
    'Old Balance Held Dest', 'Balance Held Dest', 'Transaction Hour',
    'Transaction Type_CASH_OUT', 'Transaction Type_DEBIT', 'Transaction Type_PAYMENT',
    'Transaction Type_TRANSFER', 'Conducting Manner_Cash',
    'Conducting Manner_Internet Banking', 'Conducting Manner_Mobile Banking',
    'Sex_MALE', 'Transaction Day', 'Age'
]

# Account Type Mapping (same as training)
ACCOUNT_TYPE_MAPPING = {
    'WADIAH CURRENT': 67, 'SAV INT BEARING': 46, 'SPECIAL SAVINGS': 50,
    # ... (include your full account type mapping here)
}

def preprocess_input(input_dict):
    """Convert raw input to model-ready format with proper encoding/transformations."""
    processed = {}
    
    # Numerical features (direct mapping)
    processed['Old Balance Org'] = input_dict['Old Balance Org']
    processed['New Balance Orig'] = input_dict['New Balance Orig']
    processed['Old Balance Held Dest'] = input_dict['Old Balance Held Dest']
    processed['Balance Held Dest'] = input_dict['Balance Held Dest']
    processed['Transaction Hour'] = input_dict['Transaction Hour']
    processed['Transaction Day'] = input_dict['Transaction Day']
    
    # Transformations
    processed['Amount in Birr'] = np.log1p(input_dict['Amount in Birr'])  # Log-transform
    processed['Age'] = input_dict['Age'] / 365  # Convert days to years
    
    # Account Type encoding
    processed['Account Type'] = ACCOUNT_TYPE_MAPPING.get(input_dict['Account Type'], -1)
    
    # Transaction Type one-hot encoding
    transaction_type = input_dict['Transaction Type']
    processed['Transaction Type_CASH_OUT'] = 1 if transaction_type == 'CASH-OUT' else 0
    processed['Transaction Type_DEBIT'] = 1 if transaction_type == 'DEBIT' else 0
    processed['Transaction Type_PAYMENT'] = 1 if transaction_type == 'PAYMENT' else 0
    processed['Transaction Type_TRANSFER'] = 1 if transaction_type == 'TRANSFER' else 0
    
    # Conducting Manner one-hot encoding
    conducting_manner = input_dict['Conducting Manner']
    processed['Conducting Manner_Cash'] = 1 if conducting_manner == 'Cash' else 0
    processed['Conducting Manner_Internet Banking'] = 1 if conducting_manner == 'Internet Banking' else 0
    processed['Conducting Manner_Mobile Banking'] = 1 if conducting_manner == 'Mobile Banking' else 0
    
    # Sex encoding
    processed['Sex_MALE'] = 1 if input_dict['Sex'] == 'Male' else 0
    
    return pd.DataFrame([processed], columns=FEATURE_NAMES)

# Streamlit app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Predict", "About"])

    if page == "Home":
        st.title("AML Transaction Monitoring System")
        st.write("""This application uses a machine learning model to predict the likelihood of a transaction being fraudulent. This application uses machine learning models to predict the likelihood of a transaction being fraudulent.
        You can enter the transaction details and the system will predict if the transaction is **Normal** or **Fraudulent**.""")
        st.markdown("""
        Predict fraudulent transactions using machine learning.  
        **Instructions**:  
        1. Go to the **Predict** page  
        2. Enter transaction details  
        3. Click **Predict**  
        """)

    elif page == "Predict":
        st.title("Transaction Fraud Detection")
        
        with st.form("input_form"):
            # Transaction Details
            col1, col2 = st.columns(2)
            with col1:
                transaction_type = st.selectbox(
                    "Transaction Type",
                    ['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
                )
                amount = st.number_input("Amount (Birr)", min_value=0.0)
                old_balance_org = st.number_input("Old Balance Origin", min_value=0.0)
                new_balance_orig = st.number_input("New Balance Origin", min_value=0.0)
                
            with col2:
                conducting_manner = st.selectbox(
                    "Conducting Manner",
                    ['Account to Account', 'Cash', 'Internet Banking', 'Mobile Banking']
                )
                account_type = st.selectbox("Account Type", list(ACCOUNT_TYPE_MAPPING.keys()))
                old_balance_dest = st.number_input("Old Balance Destination", min_value=0.0)
                balance_dest = st.number_input("Balance Destination", min_value=0.0)
            
            # User Details
            sex = st.radio("Sex", ["Male", "Female"])
            age_days = st.number_input("Age (in days)", min_value=0)
            transaction_hour = st.slider("Transaction Hour", 0, 23)
            transaction_day = st.selectbox("Transaction Day (0=Mon)", options=list(range(7)))
            
            submitted = st.form_submit_button("Predict Fraud Risk")
        
        if submitted:
            input_data = {
                'Transaction Type': transaction_type,
                'Amount in Birr': amount,
                'Old Balance Org': old_balance_org,
                'New Balance Orig': new_balance_orig,
                'Conducting Manner': conducting_manner,
                'Account Type': account_type,
                'Old Balance Held Dest': old_balance_dest,
                'Balance Held Dest': balance_dest,
                'Sex': sex,
                'Age': age_days,
                'Transaction Hour': transaction_hour,
                'Transaction Day': transaction_day
            }
            
            try:
                processed_df = preprocess_input(input_data)
                prediction = model.predict(processed_df)[0]
                
                # Display result
                result = "Fraud Detected!" if prediction == 1 else "Normal Transaction"
                color = "#FF4B4B" if prediction == 1 else "#00CC96"
                st.markdown(f"<h2 style='text-align: center; color: {color};'>{result}</h2>", 
                           unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

    elif page == "About":
        st.title("About the Application")
        st.write("""This application helps detect potential money laundering transactions using a trained machine learning model.
        This application was developed to help detect potential money laundering transactions using machine learning.
        The model is trained on a dataset of historical transactions and uses various transaction features to make predictions.""")
        st.write("""  
        
        **Contact**:  
        [Ameha Abera Kidane](mailto:amehaabera@gmail.com)  
        [LinkedIn Profile](https://www.linkedin.com/in/ameha-abera-kidane/)  
        [Github Profile](https://github.com/amehaabera)
        """)

if __name__ == "__main__":
    main()
