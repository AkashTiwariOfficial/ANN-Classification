import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import streamlit as st

model = tf.keras.models.load_model('model.h5')

with open("one_hot_code_Geography.pkl", 'rb') as file:
    one_hot_code_Geography=pickle.load(file)

with open("label_encode_gender.pkl", 'rb') as file:
    label_encode_gender=pickle.load(file)

with open("scaler.pkl", 'rb') as file:
    scaler=pickle.load(file)


st.title("Customer Churn Prediction")

geography=st.selectbox('Geography', one_hot_code_Geography.categories_[0])
gender=st.selectbox('Gender', label_encode_gender.classes_)
age=st.slider("Age", 18, 92)
balance=st.number_input("Balance")
credit_score=st.number_input("Credit Score")
estimated_salary=st.number_input("Estimated Salary")
tenure=st.slider("Tenure", 0, 40)
no_of_products=st.slider("Number of Products", 1, 5)
hs_cr_card=st.selectbox("Has Credit Card", [0,1])
is_active_member=st.selectbox("Is Active Member", [0,1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encode_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [no_of_products],
    'HasCrCard': [hs_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


code_Geography = one_hot_code_Geography.transform([[geography]])
new_Geo = pd.DataFrame(code_Geography.toarray(), columns=one_hot_code_Geography.get_feature_names_out(['Geography']))

input_df = pd.concat([input_data, new_Geo], axis=1)

input_scaler = scaler.transform(input_df)
prediction = model.predict(input_scaler)
prediction_Probability = prediction[0][0]


st.write(f'Churn Probability: {prediction_Probability:.2f}')

if prediction_Probability > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')