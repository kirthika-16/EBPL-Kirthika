import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib  # For saving and loading the model
import streamlit as st
import json  # To parse the JSON input

# 🚀 STEP 1: Load the CSV dataset
df = pd.read_csv("creditcard.csv")  # Replace with your CSV file path

# 🚀 STEP 2: Prepare data (remove unwanted columns, handle missing values)
df.fillna(0, inplace=True)

# 🚀 STEP 3: Separate Features and Target (Fraud Label)
X = df.drop(columns=['Class'])  # Features
y = df['Class']  # Target: 0 = Not Fraud, 1 = Fraud

# 🚀 STEP 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 🚀 STEP 5: Train AI Model (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 🚀 STEP 6: Save Model for Future Use
joblib.dump(rf_model, "fraud_detector.pkl")  # Save trained model

# 🚀 STEP 7: Load Model Once (for optimization)
model = joblib.load("fraud_detector.pkl")

# 🚀 STEP 8: Function for Real-Time Fraud Detection with UI-style Output
def detect_fraud(transaction):
    """
    Takes a new transaction (dictionary format), converts it to a DataFrame,
    and predicts whether it's fraud or not.
    """
    # Convert dictionary to DataFrame (expects same columns as training data)
    input_data = pd.DataFrame([transaction])

    # Make prediction (1 = Fraud, 0 = Not Fraud)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probability of fraud

    return prediction, probability

# 🚀 STEP 9: Create Streamlit UI
st.title('Fraud Detection System')

# Custom CSS for improving the UI
st.markdown("""
    <style>
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .result-container {
            padding: 10px;
            margin-top: 20px;
            border-radius: 8px;
        }
        .safe {
            background-color: #d4edda;
            color: #155724;
        }
        .fraud {
            background-color: #f8d7da;
            color: #721c24;
        }
        .input-area {
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# 🚀 STEP 10: Display Pie Chart for Fraud vs Non-Fraud Distribution
fraud_counts = y.value_counts()
labels = ['Not Fraud', 'Fraud']
colors = ['lightblue', 'lightcoral']

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(fraud_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
ax.set_title('Fraud vs Non-Fraud Distribution in the Dataset')
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
st.pyplot(fig)

# 🚀 STEP 11: JSON Input for New Transaction
st.subheader('Enter New Transaction Details (JSON format)')

# Provide an example of the required JSON format
example_json = '''{
    "Time": 100000,
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536346,
    "V4": 1.378155,
    "V5": -0.338320,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551600,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 149.62
}'''

# Text area for the user to input JSON data
json_input = st.text_area("Paste the Transaction JSON here:", example_json, height=250)

# Button to process the JSON input
if st.button('Check Transaction'):
    try:
        # Parse the JSON input
        new_transaction = json.loads(json_input)

        # Check if the JSON contains all necessary fields
        required_fields = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                           'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 
                           'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
        if all(field in new_transaction for field in required_fields):
            # Detect fraud using the model
            prediction, probability = detect_fraud(new_transaction)

            # Display results in a styled container
            result_class = 'fraud' if prediction == 1 else 'safe'
            st.markdown(f"""
                <div class="result-container {result_class}">
                    <h3>Prediction Result</h3>
                    <p><strong>Risk:</strong> {probability*100:.2f}%</p>
                    <p><strong>Status:</strong> {'FRAUDULENT' if prediction == 1 else 'SAFE'}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("The JSON input is missing one or more required fields.")
    
    except json.JSONDecodeError:
        st.error("Invalid JSON format. Please ensure the input is valid JSON.")
    except ValueError:
        st.error("An error occurred while processing the data. Please check the input values.")
