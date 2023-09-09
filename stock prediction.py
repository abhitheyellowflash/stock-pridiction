#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# In[3]:


# Loading historical stock price data 
data = pd.read_csv('TSLA.csv')
data['Date'] = pd.to_datetime(data['Date'])  
data.set_index('Date', inplace=True)



# In[4]:


# Selecting the 'Close' column as the target variable for prediction
data = data[['Close']]

# Normalizing the data using Min-Max scaling
scaler = MinMaxScaler()
data['Close'] = scaler.fit_transform(data[['Close']])

# Spliting the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]



# In[5]:


# Creating sequences of data for training
def create_sequences(data, sequence_length):
    sequences = []
    target = []
    for i in range(len(data) - sequence_length):
        sequence = data.iloc[i:i+sequence_length]
        target_val = data.iloc[i+sequence_length]
        sequences.append(sequence.values)
        target.append(target_val.values)
    return np.array(sequences), np.array(target)

sequence_length = 10  # You can adjust this value
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)



# In[6]:


# Defining the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.LSTM(units=50),
    tf.keras.layers.Dense(1)
])

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluating the model on the test data
test_loss = model.evaluate(X_test, y_test)



# In[8]:


# Predicting stock prices on the test data
predicted_prices = model.predict(X_test)

# Inverse transform the predicted prices to the original scale
predicted_prices = scaler.inverse_transform(predicted_prices)
y_test = scaler.inverse_transform(y_test)



# In[9]:


# Visualize the actual vs. predicted stock prices
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(predicted_prices, label='Predicted Prices', color='red')
plt.title('Stock Price Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[ ]:




