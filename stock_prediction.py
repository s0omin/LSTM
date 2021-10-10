import math
import numpy as np
import pandas_datareader as web
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def predict_closing(name, s_date, e_date) :
    # Get the stock quote
    df = web.DataReader(name, data_source='yahoo', start = s_date, end=e_date)

    # Create new dataframe with only the 'Close' column
    data = df.filter(['Close'])
    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = math.ceil(len(dataset) * 0.8)


    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set
    # Create the scaled training data set
    train_data = scaled_data[0:training_data_len, :]

    # Split the data into x_train and y_train data sets
    # x_train dataset contains the past 60 values
    # y_train dataset contains the 61st value that we want our model to predict
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i,0])

    # Convert x_train and y_train datasets to numpy arrays
    x_train,y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    # LSTM network expects the input to be 3 dimensional
    # our dataset is 2 dimensional
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer = 'adam', loss='mean_squared_error')

    # Train the model
    # batch_size = total number of training examples present in a signle batch
    # epochs = the number of iteration when entire dataset is passed forward and backward through a neural network
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing data set
    # Create a new array containing scaled values from indx 1705 to 2003
    test_data = scaled_data[training_data_len - 60 : , :]
    x_test = [] # will contain our past 60 values
    y_test = dataset[training_data_len:, :] # contains true values (starting with index 61)

    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1],1))

    # get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # evaluate our model by getting RMSE
    rmse = np.sqrt(np.mean(predictions-y_test)**2)
    print("Mean Squared Error (MSRE): ", rmse)

    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]

    valid['Predictions'] = predictions

    # Visualize the data
    plt.figure(figsize=(16,8))
    plt.title('LSTM Model')
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close','Predictions']])
    plt.legend(['Train', 'Vala', 'Predictions'], loc = 'lower right')
    plt.show()
    



predict_closing('TSLA', '2020-01-01', '2021-01-01')
