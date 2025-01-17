# Stock Price Prediction using LSTM

This project demonstrates the use of LSTM (Long Short-Term Memory) neural networks to predict stock prices. The model is trained on historical stock data, and it predicts future prices based on past patterns. It includes a Streamlit-based web interface where users can input a stock ticker and select a date range to get stock price predictions.

## Table of Contents
- [Project Description](#project-description)
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Libraries Used](#libraries-used)
- [API](#API)


## Project Description

This application allows users to:
- Predict future stock prices using historical data.
- Input a stock ticker and a date range to generate predictions.
- View the predicted prices and the actual prices for the specified date range.

The model uses LSTM (Long Short-Term Memory), a type of recurrent neural network (RNN), to process and predict stock prices based on time-series data.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/stock-price-prediction.git
   ```
2. Navigate to the project directory:
   ```
   cd stock-price-prediction
   ```
3. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run deploy.py
   ```
2. The application will open in your default web browser.
3. Enter the stock ticker symbol (e.g., AAPL, MSFT, TSLA) and select the start and end dates for the prediction.
4. Click the "Run Prediction" button to generate the stock price prediction.
5. The application will display a plot showing the actual and predicted stock prices.

## How It Works

The stock price prediction is based on a machine learning model (LSTM). The main steps are:

1. **Data Collection**: Historical stock data is collected using the `yfinance` library.
2. **Data Preprocessing**: The collected data is scaled and prepared for the LSTM model.
3. **Model Training**: The LSTM model is trained on the historical stock data to learn the patterns and predict future prices.
4. **Prediction**: The model predicts future stock prices for the selected date range.
5. **Visualization**: The results are displayed in the Streamlit app as a plot and a table.

## Libraries Used

- **TensorFlow**: For building and training the LSTM model.
- **Scikit-learn**: For data preprocessing and model evaluation.
- **Matplotlib**: For visualizing the stock prices.
- **yfinance**: For downloading historical stock data.
- **Streamlit**: For creating the interactive web interface.
- **NumPy**: For numerical operations.
- **Pandas**: For data manipulation and handling.
- 
## API

The project uses the following APIs:

- **Yahoo Finance API**: Fetches historical stock data.
- **Streamlit API**: Provides the web-based user interface for the application.


## License

This project is licensed under the [MIT License](LICENSE).
