import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
from keras.models import load_model
import tensorflow as tf
import streamlit as st

#Load model file
model = tf.keras.models.load_model('model.h5')

#Load encoder and scaler files

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_geoencoder = pickle.load(file)

with open('scalar.pkl', 'rb') as file:
    scaler = pickle.load(file)

## Stream lit

st.title("Customer churn prediction")

#userinput

geography = st.selectbox('Geography', onehot_geoencoder.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


#create dataframe

input_data = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary
}

geo_encoded = onehot_geoencoder.transform([[input_data['Geography']]]).toarray()
geo_enoded_data = pd.DataFrame(geo_encoded, columns=onehot_geoencoder.get_feature_names_out(['Geography']))

input_df = pd.DataFrame([input_data])
input_df = pd.concat([input_df, geo_enoded_data], axis=1)

input_df['Gender'] = label_encoder_gender.transform(input_df["Gender"])
input_df.drop(columns=['Geography'], inplace=True)

print("input_df\n",input_df)

# Scale the input data
input_df_scaled = scaler.transform(input_df)


# Predict churn
prediction = model.predict(input_df_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')




