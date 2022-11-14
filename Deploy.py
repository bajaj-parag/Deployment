# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:48:00 2022

@author: Dell
"""

import streamlit as st
import pandas as pd
import matplotlib
from sklearn.linear_model import LogisticRegression
import numpy as np

st.title( 'Model Deployment: Logistic Regression')
st.sidebar.header( 'User Input Parameters')

def user_input_features():
    
    CLMSEX = st.sidebar.selectbox('Gender',( '1 ', '0'))
    CLMINSUR= st.sidebar.selectbox('Insurance ' , ( '1 ' ,'0'))
    SEATBELT = st.sidebar.selectbox('SeatBelt ' , ( '1','0'))
    CLMAGE = st.sidebar.number_input("Insert the Age")
    LOSS =  st.sidebar.number_input("Insert Loss")
    data = {'CLMSEX':CLMSEX,
            'CLMINSUR':CLMINSUR,
            'SEATBELT':SEATBELT,
            'CLMAGE':CLMAGE,
            'LOSS':LOSS}
    features = pd.DataFrame(data,index =[0])
    return features

df = user_input_features()
st.subheader('User Input Paramters')
st.write(df)

claimants = pd.read_csv('claimants.csv')
claimants.drop(['CASENUM'], inplace=True, axis= 1)
claimants = claimants.dropna()

X = claimants.iloc[:,[1,2,3,4,5]]
Y = claimants.iloc[:,0]
clf = LogisticRegression()
clf.fit()

prediction = clf.predict(df)
prediction_proba= clf.predict_proba(df)

st.subheader('Predict Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'NO')

st.subheader('Prediction Probability')

st.write(prediction_proba)
st.write(prediction)










