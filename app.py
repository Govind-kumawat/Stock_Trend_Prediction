import numpy  as np
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yfin
from keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st


st.title('Stock Trend Prediction')


yfin.pdr_override()
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

st.subheader('Data from 2010-2022')
df = pdr.get_data_yahoo(user_input, start='2010-01-01', end='2022-12-31')

st.write(df.describe())

# Visualize the Plot

st.subheader('Closing Price VS Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

st.subheader('Closing Price VS Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

st.subheader('Closing Price VS Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100)
plt.plot(ma200)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

#Splitting data into training and testing set

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
data_training_array = scaler.fit_transform(data_training)

#Load the model
model = load_model('Keras_model.h5')

past_100days = data_training.tail(100)
frames = [past_100days, data_testing]
final_df = pd.concat(frames)
input_data = scaler.fit_transform(final_df)

#Testing the model

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100 : i])
  y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test) 


#Making Predictions

y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]

y_test = y_test * scale_factor
y_predicted = y_predicted * scale_factor

#Visualize the final Graph

st.subheader('Predictions vs Original Price')
fig2 = plt.figure(figsize = (12, 6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
