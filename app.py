import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load trained model
model = tf.keras.models.load_model('model.h5')

# Load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title("Customer Churn Prediction App")

# Input fields with unique keys
geography = st.selectbox('Geography', one_hot_encoder.categories_[0], key="geography")
gender = st.selectbox('Gender', label_encoder_gender.classes_, key="gender")
age = st.slider('Age', 18, 92, key="age")
balance = st.number_input('Credit Balance', key="balance")
credit_score = st.number_input('Credit Score', key="credit_score")
estimated_salary = st.number_input('Estimated Salary', key="estimated_salary")
tenure = st.slider('Tenure', 1, 10, key="tenure")  # Adjusted max range for more flexibility
num_of_products = st.slider('Number of Products', 1, 4, key="num_of_products")
has_cr_card = st.selectbox('Has Credit Card', [0, 1], key="has_cr_card")
is_active_member = st.selectbox('Is Active Member', [0, 1], key="is_active_member")

# Prepare input data dictionary (matching feature names exactly)
input_data = ({
    'CreditScore': [credit_score],  # Matches training data
    'Gender': [label_encoder_gender.transform([gender])[0]],  # Matches training data
    'Age': [age],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],  # Matches training data
    'EstimatedSalary': [estimated_salary],  # Matches training data
    'Tenure': [tenure],
    'HasCrCard': [has_cr_card],  # Matches training data
    'IsActiveMember': [is_active_member]  # Matches training data
})

# Encode geography and transform to DataFrame
geo_encoded = one_hot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder.get_feature_names_out(['Geography']))

# Convert input_data to a DataFrame before concatenation
input_data_df = pd.DataFrame(input_data)

# Concatenate DataFrames properly
data = pd.concat([input_data_df.reset_index(drop=True), geo_encoded_df], axis=1)

# Ensure column order matches training data
expected_columns = scaler.feature_names_in_  # Get feature names from fitted scaler
data = data[expected_columns]  # Reorder columns to match training data

# Scale input data
input_data_scaled = scaler.transform(data)

# Prediction
prediction = model.predict(input_data_scaled)
prediction_probab = prediction[0][0]

# Display results
st.write(f'Churn Probability: {prediction_probab:.2f}')

if prediction_probab > 0.5:
    st.write("The Customer is likely to churn.")
else:
    st.write("The Customer is not likely to churn.")
