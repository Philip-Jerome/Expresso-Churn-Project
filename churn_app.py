import streamlit as st
import joblib
import pandas as pd
import numpy as np


churn_model = joblib.load('churn_predictor.joblib')
top_pack_encoder = joblib.load('job_pack_encoder.joblib')
region_encoder = joblib.load('region_encoder.joblib')


st.title('Supervised Machine Learning')
st.header('Expresso Churn Prediction App')

column_desc = pd.read_csv('VariableDefinitions.csv',header=1, names= ['Features', 'French', 'English'])
column_desc = column_desc.iloc[1:19]
st.write("Expresso is an African telecommunications services company that provides telecommunication services in two African markets: Mauritania and Senegal. The data describes 2.5 million Expresso clients with more than 15 behaviour variables in order to predict the clients' churn probability.")
st.dataframe(column_desc)



with st.sidebar:
    region = st.selectbox('What region is the customer?', [i for i in region_encoder.classes_])
    tenure = st.number_input('How long has the customer spent transacting with the company', 1,24)
    montant  = st.number_input('Top-up amount',50,235500)
    freq_recharge = st.number_input('A number of times the customer refilled',1,113)
    revenue = st.number_input('Monthly income of each client', 1,221999)
    arpu_segment = st.number_input('Income over 90 days / 3',0,74000)
    freq = st.number_input('Number of times the client has made an income',1,91)
    data_vol = st.number_input('Number of connections', 0,926547)
    on_net = st.number_input('Inter expresso call', 0,50809)
    orange = st.number_input('Call to orange', 0, 6429)
    tigo = st.number_input('Call to tigo', 0,2899)
    zone1 = st.number_input('Call to zone 1', 0,1867)
    zone2 = st.number_input('Call to zone2', 0,1346)
    regularity = st.number_input('Number of times the client is active for 90 days',1,62)
    top_pack = st.selectbox('The most active packs', [i for i in top_pack_encoder.classes_])
    freq_top_pack = st.number_input('Number of times the client has activated the top pack packages',1,592)


# # Prediction
features = [region_encoder.transform([region]),
            tenure,
            montant,
            freq_recharge,
            revenue,
            arpu_segment,
            freq,
            data_vol,
            on_net,
            orange,
            tigo,
            zone1,
            zone2,
            regularity,
            top_pack_encoder.transform([top_pack]),
            freq_top_pack
            ]

user_data = pd.DataFrame([features], columns=[ 'REGION', 'TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE',
        'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO',
    'ZONE1', 'ZONE2', 'REGULARITY', 'TOP_PACK', 'FREQ_TOP_PACK'])

# st.write('User DataFrame')
# st.dataframe(user_data)



if st.sidebar.button('Predict'):
    st.dataframe(user_data)
    prediction = churn_model.predict(user_data)
    if prediction == 0:
        st.write('Client is retained')
    else:
        st.write('Client is Lkely to Churn')