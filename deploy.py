import streamlit as st
import datetime as dt
import matplotlib.pyplot as plt
from stock_prediction import predict_stock_price  

def main():
    st.title("Stock Price Prediction using LSTM")

    stock_ticker = st.text_input('Enter Stock Ticker (e.g., AAPL, MSFT, TSLA):', 'AAPL')
    start_date = st.date_input('Select Start Date:', dt.datetime(2024, 1, 1))
    end_date = st.date_input('Select End Date:', dt.datetime(2026, 10, 13))

    if st.button('Run Prediction'):
        
        predicted_prices, actual_prices = predict_stock_price(stock_ticker, start_date, end_date)

        fig, ax = plt.subplots()
        ax.plot(actual_prices, color='black', label=f'Actual {stock_ticker} price')
        ax.plot(predicted_prices, color='green', label=f'Predicted {stock_ticker} price')
        ax.set_title(f'{stock_ticker} Share Price Prediction')
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{stock_ticker} Share Price')
        ax.legend()

        st.pyplot(fig)

if __name__ == '__main__':
    main()
