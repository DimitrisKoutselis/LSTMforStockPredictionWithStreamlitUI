import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

app = FastAPI()


class StockRequest(BaseModel):
    stock_name: str


STOCK_FILE_PATHS = {
    'TSLA': 'data/TSLA_data.csv',
    'AAPL': 'data/AAPL_data.csv',
    'AMZN': 'data/AMZN_data.csv',
    'MSFT': 'data/MSFT_data.csv',
    'GOOG': 'data/GOOG_data.csv',
    'SPY': 'data/SPY_data.csv',
    'GC=F': 'data/GC=F_data.csv',
    'user0': 'data/user0_data.csv',
    'user1': 'data/user0_data.csv',
    'user2': 'data/user0_data.csv',
    'user3': 'data/user0_data.csv',
    'user4': 'data/user0_data.csv',
    'user5': 'data/user0_data.csv',
    'user6': 'data/user0_data.csv',
    'user7': 'data/user0_data.csv',
    'user8': 'data/user0_data.csv',
    'user9': 'data/user0_data.csv',
}


@app.post('/LSTM_Predict')
async def predict(stock_request: StockRequest):
    stock_name = stock_request.stock_name
    try:
        file_path = STOCK_FILE_PATHS[stock_name]
        df = pd.read_csv(file_path)
    except KeyError:
        raise HTTPException(status_code=422, detail='Invalid stock name')
    
    data = df.filter(['Close'])

    dataset = data.values

    training_data_len = int(np.ceil(len(dataset) * .8))
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[:training_data_len, :]
    seq_length = 60
    x_train = []
    y_train = []

    for i in range(seq_length, len(train_data)):
        x_train.append(train_data[i-seq_length:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = keras.Sequential([
        keras.layers.LSTM(512, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(256, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(128, return_sequences=False),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    weights_file = f'weights/{stock_name}_weights.weights.h5'
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    if os.path.exists(weights_file):
        model.load_weights(weights_file)
    else:
        history = model.fit(x_train, y_train, batch_size=32, epochs=125)
        if not stock_name.startswith("user"):
            model.save_weights(f'weights/{stock_name}_weights.weights.h5')

    test_data = scaled_data[training_data_len-seq_length:, :]

    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(seq_length, len(test_data)):
        x_test.append(test_data[i-seq_length:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    predict_price = [price[0] for price in predictions.tolist()]

    for price in predictions.tolist():
        predict_price.append(price[0])

    return {'prediction': predict_price}