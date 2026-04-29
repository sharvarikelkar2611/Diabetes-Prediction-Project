import streamlit as st
import pandas as pd
import pickle as pkl

sc = pkl.load(open('scaler.pkl', 'rb+'))
reg = pkl.load(open('DPP.pkl', 'rb+'))


st.title("Diabetic Patient Prediction Project")
gender = st.selectbox("Select Gender", ['Female', 'Male', 'Other'])
age = st.number_input("Enter Age", min_value=0, max_value=100, value=50)
hypertension = st.selectbox("Select hypertension", ['Yes','No'])
heart_disease = st.selectbox("Select Heart Disease", ['Yes','No'])
smoking_history = st.selectbox("Select Smoking History", ['never', 'No Info', 'former', 'not current', 'ever', 'current'])
bmi = st.number_input("Enter BMI", min_value=20, max_value=50, value=28)
HbA1c_level = st.number_input("Enter HbA1c Level", min_value=3.0, max_value=10.0, value=6.6, step=0.1)
blood_glucose_level = st.number_input("Enter Blood Glucose Level", min_value=50, max_value=500, value=200)
if gender == 'Female':
    gender = 0
elif gender == 'Male':
    gender = 1
else:
    gender = 2

if smoking_history == 'never':
    smoking_history = 0
elif smoking_history == 'No Info':
    smoking_history = 1
elif smoking_history == 'former' or smoking_history == 'not current':
    smoking_history = 2
elif smoking_history == 'ever':
    smoking_history = 3
else:
    smoking_history = 4

if hypertension == 'Yes':
    hypertension = 1
else:
    hypertension = 0

if heart_disease == 'Yes':
    heart_disease = 1
else:
    heart_disease = 0

if st.button("Predict"):
    myinput=[[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]]
    columns=['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
       'bmi', 'HbA1c_level', 'blood_glucose_level']
    datainput=pd.DataFrame(data=myinput, columns=columns)
    datainput_scaled=sc.transform(datainput)
    result=reg.predict(datainput_scaled)
    
    if result == 0:
        st.success("Person is not diabetic")
    else:
        st.error("Person is diabetic")
        




