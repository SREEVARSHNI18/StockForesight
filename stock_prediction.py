import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential,Dense
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import train_test_split

# Fetch data from Yahoo Finance
def fetch_data(stock_ticker, start_date, end_date):
    df = yf.download(stock_ticker, start=start_date, end=end_date)
    return df

def preprocess_data(df):
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler


def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_lstm_data(data, look_back=60):
    x_data, y_data = [], []
    for i in range(look_back, len(data)):
        x_data.append(data[i-look_back:i, 0])
        y_data.append(data[i, 0])
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    
    return x_data, y_data

def predict_stock_price(stock_ticker, start_date, end_date):
    df = fetch_data(stock_ticker, start_date, end_date)
    scaled_data, scaler = preprocess_data(df)
    
    #training
    x_data, y_data = prepare_lstm_data(scaled_data)
    
    # Split into training and test data
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=False)
   
    model = create_lstm_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    
    #  predictions on the test data
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    return predicted_prices, actual_prices
