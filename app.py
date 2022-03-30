import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

used_cars_data = pd.read_csv('used_cars_data.csv')
data_sans_unimp = used_cars_data.drop(['Model'],axis=1)
data_sans_missingval = data_sans_unimp.dropna(axis=0)
one_p_mark = data_sans_missingval['Price'].quantile(0.99)
data_1 = data_sans_missingval[data_sans_missingval['Price']<one_p_mark]
one_p_mark = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<one_p_mark]
data_3 = data_2[data_2['EngineV']<6.5]
old_one_p = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>old_one_p]
data_cleaned = data_4.reset_index(drop=True)
log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price
data_cleaned = data_cleaned.drop(['Price'],axis=1)
data = data_cleaned
le = LabelEncoder()
data['Brand_encoding'] = le.fit_transform(data['Brand'])
data['Body_encoding'] = le.fit_transform(data['Body'])
data['Engine_Type_encoding'] = le.fit_transform(data['Engine Type'])
data['Reg_enc'] = le.fit_transform(data['Registration'])
data_preprocessed = data.drop(['Brand', 'Body', 'Engine Type', 'Registration'], axis=1)
targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'],axis=1)
scaler = StandardScaler()
scaler.fit(inputs)
scaled_inputs = scaler.transform(inputs)
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, test_size=0.2, random_state=365)
rf = RandomForestRegressor()
rf.fit(x_train,y_train)

brands = ['Audi', 'BMW', 'Mercedes-Benz', 'Mitsubishi', 'Renault', 'Toyota', 'Volkswagen']
brand_dict = {'Audi': 0, 'BMW': 1, 'Mercedes-Benz': 2, 'Mitsubishi': 3, 'Renault': 4, 'Toyota': 5, 'Volkswagen': 6}

bodies = ['Crossover', 'Hatch', 'Other', 'Sedan', 'Vagon', 'Van']
body_dict = {'Crossover': 0, 'Hatch': 1, 'Other': 2, 'Sedan': 3, 'Vagon': 4, 'Van': 5}

engine_types = ['Diesel', 'Gas', 'Other', 'Petrol']
engine_type_dict = {'Diesel': 0, 'Gas': 1, 'Other': 2, 'Petrol': 3}

registration = ['Yes', 'No']
registration_dict = {'No': 0, 'Yes': 1}



st.set_page_config(page_title='Car Bazaar')
st.markdown("<h1 style='font-family:georgia; text-align: center; color:#D42F0B' > Car Bazaar </h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'> Used Cars Price Estimation using Machine Learning </h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'> For best results, provide real and accurate inputs </p>", unsafe_allow_html=True)

st.markdown("<h3> Car Type </h3>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

user_brand = col1.selectbox(label='Brand', options=brands)
brand = brand_dict[user_brand]

user_body_type = col2.selectbox(label='Body Type', options=bodies, help='Built of the car')
body_type = body_dict[user_body_type]

st.markdown("<h3> Car Age and Usage </h3>", unsafe_allow_html=True)
col3, col4 = st.columns(2)

year = col3.slider('Year of Manufacture', 1980,2020,2005)
mileage = col4.number_input(label='Mileage on the Car', min_value=0.00, help='Number of MILES')

st.markdown("<h3> Engine Specs </h3>", unsafe_allow_html=True)
col5, col6 = st.columns(2)

user_engine_type = col5.selectbox(label='Engine type',options=engine_types, help='Type of Fuel used (Gas, Petrol)')
engine_type = engine_type_dict[user_engine_type]

engineV = float(col6.number_input(label='Engine Volume', max_value=6.51, min_value=1.00,help='You can google if you don\'t have any idea'))

st.markdown("<h3> Registration </h3>", unsafe_allow_html=True)
col7, col8, col9 = st.columns(3)

user_reg = col8.selectbox(label='Is the car Registered?', options=registration)
reg = registration_dict[user_reg]

user_input = np.array([[mileage, engineV, year, brand, body_type, engine_type, reg]])
user_input_scaled = scaler.transform(user_input)

predict = col8.button('Estimate')

if predict:
    try:
        y_pred = rf.predict(user_input_scaled)
        y_pred = round(float(np.exp(y_pred)))
        out = 'Based on 4000+ real-world sales, the estimated price for above car is $ '+ str(y_pred) + '.'
        st.success(out)
    except:
        st.error('Something is wrong with the inputs, try again!')

st.header('About')
about = """
            This Machine Learning application uses a Random Forest Regressor to estimate the appropriate price for a used car based on past deals and sales.
            The app handles 7 brands.
            The dataset used to train the model can be found here - https://github.com/Abhishek-Dxt/Car-Bazaar/blob/master/used_cars_data.csv \n
            The entire project can be accessed on my GitHub - https://github.com/Abhishek-Dxt/Car-Bazaar \n
            Check my other projects and contact details at - https://abhishek-dxt.github.io/
"""
st.write(about)