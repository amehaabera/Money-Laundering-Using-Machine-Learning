import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the saved model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Check the type of the loaded model
st.write(f"Loaded model type: {type(model)}")  # Debug: Validate model loading

# Define feature names (ensure this matches the training order)
feature_names = [
    'Transaction Type', 'Conducting Manner', 'Old Balance Org',
    'New Balance Orig', 'Amount in Birr', 'Sex', 'Account Type',
    'Old Balance Held Dest', 'Balance Held Dest',
    'Transaction Hour', 'Transaction Day', 'Age'
]

# Transaction Type Mapping
transaction_type_mapping = {
    'CASH-IN': 0, 'CASH-OUT': 1, 'DEBIT': 2,
    'PAYMENT': 3, 'TRANSFER': 4
}

# Conducting Manner Mapping (Updated)
conducting_manner_mapping = {
    'Account to Account': 0,
    'Cash': 1,
    'Internet Banking': 2,
    'Mobile Banking': 3
}

# Account Type Mapping based on the encoder (you should use the same mapping that was used during training)
account_type_mapping = {
    'WADIAH CURRENT': 67, 'SAV INT BEARING': 46, 'SPECIAL SAVINGS': 50, 'WADIAH SAVING': 68,
    'CURRENT ACCOUNT': 7, 'Fitayah Savings': 19, 'Wadiah Special': 75, 'APOLLO': 4, 
    'ADEY SAVING': 0, 'FCY Prepaid tra': 18, 'Zahrah Savings': 78, 'AFLA SAVING': 1, 
    'Musnin Savings': 37, 'BALEWUL SAVING': 6, 'Foreign cur sav': 20, 'MUDAY ABYS': 35, 
    'HAJJ SAVING ACC': 22, 'Spec.Large IQUB': 60, 'AMEEN GOAL SAVI': 3, 'Wadiah Muday Sa': 73,
    'STAFF ORDI SAVI': 53, 'WADIHA IQUUB': 70, 'MIN SAV INT BEA': 29, 'Large IQUB': 26,
    'ASRAT BEKURAT': 5, 'Wadiah Educatio': 72, 'EDUCATION SAV': 16, 'NON-RES FCY AC': 38,
    'AMEEN GOAL S -M': 2, 'SAVING PLUS': 48, 'STAFF WADIYA': 55, 'Medium IQUB': 36, 
    'Small Size IQUB': 59, 'WADIAH YAD DHAM': 69, 'DD NR FCY- DIAS': 8, 'SALARY ADV': 45, 
    'MUDARABA SAVING': 33, 'IDDR a': 24, 'ZEKAT WADIA SAV': 76, 'PROV EMP CON': 42, 
    'Staff Women': 61, 'MUDARABA ZAHRAH': 34, 'MIN SAV IFB': 28, 'SAV NON-INT': 47, 
    'STAFF SPEC SAVI': 54, 'NRT Account': 40, 'NRNT ACCOUNT': 39, 'MORTGAGE.LN': 30,
    'Private Pens': 43, 'SPEC SAVI DEPO': 49, 'Ret Main a/c': 44, 'ECX PAYO-CLIENT': 12,
    'VEHICLE.LNS': 65, 'Saving Diaspora': 58, 'PL.CONSU.LOAN': 41, 'ECX PAYIN CLIEN': 11,
    'GOV PENSION': 21, 'STAFF.MORG.LN': 57, 'ZEKAT WADIAH SA': 77, 'MUDARABA MUSNIN': 32,
    'MUDARABA FITAYA': 31, 'TEEN YOUTH SAV': 62, 'ECX payin non-m': 14, 'STAFF ZEHA SAVI': 56,
    'STAF.PL.CNSULN': 52, 'TENORED.WC.SME': 64, 'DIAS MORTGAGE': 10, 'EXP.ADV.SAL.COR': 17,
    'HOT&SER.PR.SME': 23, 'TENORED.WC.COR': 63, 'WAD SAV- FCY AC': 66, 'Wadiah Diaspora': 71, 
    'DIAS MORTG SAV': 9, 'Loan Related Ex': 27, 'ECX payin MEMB': 13, 'Investment H/U': 25,
    'ECXpayout non-m': 15, 'Wadiah Retentio': 74, 'SPL.PUR.VEH.SME': 51
}

# Define preprocessing to match training transformations
def preprocess_data(data):
    """Preprocess input data to match the trained model's format."""
    input_df = data.copy()
    
    # Initialize DataFrame for preprocessed data
    preprocessed_data = pd.DataFrame(columns=feature_names)

    # Map categorical values to numeric encoding
    preprocessed_data['Transaction Type'] = input_df['Transaction Type'].map(transaction_type_mapping)
    preprocessed_data['Conducting Manner'] = input_df['Conducting Manner'].map(conducting_manner_mapping)  # Updated line
    preprocessed_data['Old Balance Org'] = input_df['Old Balance Org']
    preprocessed_data['New Balance Orig'] = input_df['New Balance Orig']
    preprocessed_data['Amount in Birr'] = np.log1p(input_df['Amount in Birr'])  # Log-transform 'Amount in Birr'
    preprocessed_data['Sex'] = input_df['Sex']
    
    # Encode Account Type using pre-defined mapping
    preprocessed_data['Account Type'] = input_df['Account Type'].map(account_type_mapping)
    
    preprocessed_data['Old Balance Held Dest'] = input_df['Old Balance Held Dest']
    preprocessed_data['Balance Held Dest'] = input_df['Balance Held Dest']
    preprocessed_data['Transaction Hour'] = input_df['Transaction Hour']
    preprocessed_data['Transaction Day'] = input_df['Transaction Day']
    preprocessed_data['Age'] = input_df['Age'] / 365  # Convert days to years

    return preprocessed_data

# Make predictions using the loaded model
def predict(data):
    preprocessed_data = preprocess_data(data)
    predictions = model.predict(preprocessed_data)
    return predictions

# Streamlit app
def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home", "App", "About"])

    if selection == "Home":
        st.title("Anti-money laundering (AML) Transaction Monitoring System")
        st.write("""This application uses a machine learning model to predict the likelihood of a transaction being fraudulent. This application uses machine learning models to predict the likelihood of a transaction being fraudulent.
        You can enter the transaction details and the system will predict if the transaction is **Normal** or **Fraudulent**.""")

    elif selection == "App":
        st.title("**Money Laundering Transaction Detection System**")

        # Input form
        st.header('Enter transaction details:')
        transaction_type = st.selectbox('Transaction Type', list(transaction_type_mapping.keys()))
        conducting_manner = st.selectbox('Conducting Manner', list(conducting_manner_mapping.keys()))  # Updated line
        old_balance_org = st.number_input('Old Balance Org')
        new_balance_orig = st.number_input('New Balance Orig', value=old_balance_org)
        amount_in_birr = st.number_input('Amount in Birr', min_value=0.0)
        sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
        account_type = st.selectbox('Account Type', list(account_type_mapping.keys()))
        old_balance_held_dest = st.number_input('Old Balance Held Dest')
        balance_held_dest = st.number_input('Balance Held Dest', value=old_balance_held_dest)
        transaction_hour = st.number_input('Transaction Hour (0-23)', min_value=0, max_value=23)
        transaction_day = st.number_input('Transaction Day (0=Monday, 6=Sunday)', min_value=0, max_value=6)
        age = st.number_input('Age (in days)', min_value=0)

        # Create DataFrame for input
        input_data = pd.DataFrame({
            'Transaction Type': [transaction_type],
            'Conducting Manner': [conducting_manner],
            'Old Balance Org': [old_balance_org],
            'New Balance Orig': [new_balance_orig],
            'Amount in Birr': [amount_in_birr],
            'Sex': [sex],
            'Account Type': [account_type],
            'Old Balance Held Dest': [old_balance_held_dest],
            'Balance Held Dest': [balance_held_dest],
            'Transaction Hour': [transaction_hour],
            'Transaction Day': [transaction_day],
            'Age': [age]
        }, columns=feature_names)

        # Predict button
        if st.button('Predict'):
            try:
                predictions = predict(input_data)
                is_fraud = 'Fraud' if predictions[0] == 1 else 'Normal'

                # Display result
                color = 'red' if is_fraud == 'Fraud' else 'green'
                st.markdown(f"<h3 style='color:{color}; font-weight:bold;'>Predicted Class: {is_fraud}</h3>", unsafe_allow_html=True)

            except ValueError:
                st.error("Please enter valid values for all fields.")
            except Exception as e:
                st.error(f"Error in prediction: {e}")

    elif selection == "About":
        st.title("About the Application")
        st.write("""This application helps detect potential money laundering transactions using a trained machine learning model.
        This application was developed to help detect potential money laundering transactions using machine learning.
        The model is trained on a dataset of historical transactions and uses various transaction features to make predictions.""")
        st.write(""" 
        ### Developer Contact:
        - **Developer**: Ameha Abera Kidane
        - **Email**: amehaabera@gmail.com
        - **Linkedin**: https://www.linkedin.com/in/ameha-abera-kidane/
        - **GitHub**: https://github.com/amehaabera
        """)

if __name__ == '__main__':
    main()
