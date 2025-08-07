# LSTM Stock Price Predictor

## Project Overview
This project demonstrates a complete, end-to-end pipeline for predicting the daily closing price of the SPDR S&P 500 ETF (SPY) using a Long Short-Term Memory (LSTM) neural network. The entire workflow is implemented in a single, well-documented Jupyter Notebook - from raw data acquisition to model evaluation and forecasting.

The primary objective is to harness historical market data to create a reliable predictive model. Through extensive feature engineering, over 50 technical and statistical indicators are generated to help the LSTM model learn temporal market dynamics.

## Key Features
- **End-to-End Workflow:** from data collection and preprocessing to modelling and predicting
- **Extensive Feature Engineering:** includes SMAs, EMAs, RSI, MACD, Bollinger Bands, Hurst Exponent, Shannon Entropy, and more.
- **Systematic Feature Selection:** reduces multicollinearity and retains the most relevant features
- **LSTM for Sequential Data:** ideal for capturing temporal patterns in time-series financial data
- **Robust Evaluation:** uses MSE, MAE, R-squared, and diagnostic plots for comprehensive performance analysis

## Methodology
1. **Data Collection & Preprocessing:**
- Pulls historical daily OHLCV data for SPY using the Alpaca API

2. **Exploratory Data Analysis (EDA):**
- Visualises price movements, trading volume, trends, and anomalies
- Highlights volatility clusters and macroeconomic impacts
  
3. **Feature Engineering:**
- A comprehensive set of features:
    - **price-based:** lagged prices, price differences, and ratios
    - **technical indicators:** SMAs, EMAs, RSI, MACD, Bollinger Bands, Momentum
    - **volatility & volume:** rolling standard deviations, rolling means
    - **volume-based:** VWAP, lagged volume, volume oscillators
    - **statistical measures:** Shannon entropy, Hurst exponent, autocorrelation
 
4. **Systematic Feature Selection:**
- Uses Random Forest to select features based on importance

5. **LSTM Model Architecture & Training:**
- The data is scaled and transformed into sequences suitable for the LSTM network
- An LSTM model is constructed using Keras with multiple layers to learn hierarchical patterns
- The model is trained on the historical data to predict the next day's percentage return

6. **Evaluation & Prediction:**
- The trained model's predictions are compared against the actual test data
- Performance is rigorously measured, and diagnostic plots are generated to visualise the accurate predictions
- The notebook concludes by using the trained model to predict the next trading day's closing price

## Getting Started
Follow these instructions to set up and run the project on your local machine.

### Prerequisites
- Python 3.9 or higher
- Pip package manager
- An active Alpaca account for API keys

### Installation & Setup
1. Clone the repository:
```bash
git clone https://github.com/KHalid-Hajeer/lstm-stock-prediction.git
cd lstm-stock-prediction
```

2. Create and activate a venv (recommended)
```bash
python -m venv venv

source venv/bin/activate
# On Windows, use:
venv\Scripts\activate
```

3. Install the required libraries from requirements.txt
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- create a file named .env in the root directory of the project
- add your Alpaca API keys to this file:
```bash
APCA_API_KEY_ID="YOUR_API_KEY"
APCA_API_SECRET_KEY="YOUR_SECRET_KEY"
```

### Usage
Once the setup is complete, you can run the notebook:
1. Launch Jupyter Notebook or JupyterLab
2. Open the lstm_stock_model.ipynb file and run the cells sequentially.

## Results
The model's performance is evaluated on an unseen test set. The final output of the notebook provides a clear prediction for the next day's closing price, along with the predicted return.
**Example Output:**
```
--- Prediction for Next Closing Day for SPY ---
Last Actual Close Price: $627.97
Predicted Daily Return: 0.40%
Predicted Next Close Price: $630.48
```

