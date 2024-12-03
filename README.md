
# Stock Price Prediction with LSTM and Twitter Sentiment Analysis

This project integrates **historical stock price data** and **Twitter sentiment analysis** to predict stock price movements using an **LSTM (Long Short-Term Memory)** neural network. By combining price trends and public sentiment, the model aims to provide actionable insights for trading strategies.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Design](#model-design)
- [Results](#results)
- [Trading Strategy](#trading-strategy)
- [How to Run](#how-to-run)
- [Future Improvements](#future-improvements)

---

## Introduction

Stock price prediction is a complex task due to the inherent volatility of financial markets. This project tackles the challenge by:
- Leveraging **LSTM neural networks** to model sequential dependencies in time-series data.
- Incorporating **Twitter sentiment analysis** to account for public perception and market sentiment.

---

## Features

- **LSTM-based Sequential Prediction**: Captures temporal patterns in stock prices.
- **Sentiment Analysis**: Adds context from tweets related to the target stock using a **BERT model** for sentiment scoring.
- **Dynamic Trading Strategy**: Implements stop-loss and take-profit mechanisms for simulated trading.

---

## Dataset

1. **Stock Price Data**: Sourced from Yahoo Finance or similar APIs.
2. **Twitter Sentiment Data**:
   - Tweets retrieved using keyword filtering for the stock ticker.
   - Sentiment scores generated using **BERT**.

---

## Methodology

### Data Preprocessing
- **Scaling**: Stock prices are normalized with MinMaxScaler.
- **Sentiment Aggregation**: Weekly sentiment scores are computed for alignment with stock price data.
- **Sequence Generation**: Data is formatted into sequences for LSTM input, capturing historical trends.

### Model Design
- **LSTM Architecture**: 
  - Sequential layers designed to process time-series data.
  - Dropout layers added to mitigate overfitting.
- **Loss Function**: Mean Squared Error (MSE) used for optimization.
- **Output**: Predicted stock price movements for the next time step.

---

## Results

- The model successfully captures trends and improves predictive accuracy when sentiment data is included.
- A trading simulation showed potential for profit when combined with a well-defined strategy.

---

## Trading Strategy

The trading strategy involves:
1. **Signals**: Buy/Sell signals derived from predicted price changes.
2. **Risk Management**:
   - **Stop-Loss**: Cuts losses when the price moves against the prediction by a defined percentage.
   - **Take-Profit**: Locks in profits when the price reaches a favorable level.

---

## How to Run

### Prerequisites
- Python 3.8+
- Required libraries: `tensorflow`, `pandas`, `nltk`, `matplotlib`, `scikit-learn`, `transformers`

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/stock-prediction-lstm-sentiment.git
   cd stock-prediction-lstm-sentiment
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook
   ```
4. Follow the cells to preprocess data, train the model, and evaluate results.

---

## Future Improvements

- **Enhanced Sentiment Analysis**:
  - Incorporate additional sentiment models or ensembles for better generalization.
- **Real-Time Data Integration**:
  - Stream live tweets and stock prices for real-time predictions.
- **Hyperparameter Optimization**:
  - Use automated tools like Optuna for fine-tuning model performance.

---
