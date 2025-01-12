import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('Traffic_Count_Study_Area.csv')
# df.head()
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())
df = df.rename(columns={'DATE':'inn'})
print(df)
print(df.shape)
df= df.dropna()
df = df.drop_duplicates()
print(df.shape)
# print(df.shape)convert to date time format
df['inn'] = pd.to_datetime(df['inn'])
weather_data = pd.read_csv('seattle-weather.csv')
weather_data = weather_data.drop_duplicates()
weather_data = weather_data.dropna()
weather_data['date'] = pd.to_datetime(weather_data['date'])
weather_data = weather_data.set_index('date')
merged_data = df.merge(weather_data, left_index=True, right_index=True)
merged_data.fillna(method='ffill', inplace = True)
merged_data.head()
merged_data = merged_data.dropna()
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(merged_data)
# CONVERT BACK TO DATAFRAME FOR EASIER HANDELING
scaled_data = pd.df(scaled_data, columns= merged_data.columns, index = merged_data.index)
# DATA SPLITTING
x = scaled_data.drop(['TOTAL VOLUME'],axis =1).values
y = scaled_data['TOTAL VOLUME'].values
# reshape to x for lstm input{samples,timestep,features}
X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
# train
train_size = int(len(X) *0.8)
X_train, X_test = X_reshaped[:train_size], X_reshaped[train_size:]
y_train, y_test = y[:train_size],y[train_size:]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout
# creare lstm model
model = Sequential([LSTM(50, return_sequences = True, input_shape=(X_train.shape[1],X_train.shape[2])),
                    Dropout(0.2),
                    LSTM(50),
                    Dropout(0.2),
                    Dense(25),
                    Dense(1)
                    ])
model.compile(optimizer ='admin', loss = 'mean_squared_error')
# train model
history = model.fit(X_train, y_train, epochs = 50, batch_size=32,validation_data=(X_yest, y_test), verbose=2)
# model evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error
# prediction
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print(f"mean absolute error(MAE):{mae}")
print(f"root mean squared error (RMSE):{rmse}")
# visualization
plt.figure(figsize=(12,6))
plt.plot(y_test, label='Actual Traffic Flow')
plt.plot(y_pred, label = 'Predicted traffic flow')
plt.tite('traaffic flow predictions')
plt.xlabel('time steps')
plt.ylabel('traffic flow')
plt.legend()
plt.show()