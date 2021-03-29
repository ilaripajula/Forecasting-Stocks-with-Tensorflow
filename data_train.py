import yahoo_fin.stock_info as si
import ta
import os
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from WindowGenerator import *

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

ticker = "NVDA"
start_date = "01/01/2015"
end_date = "03/27/2021"
index_as_date = True
interval = "1d"

# =============================================================================
# IN_STEPS is the length of training data
# OUT_STEPS outsteps is the size of label predicitions we want to make
# When adjusting these values one needs to keep in mind the amount of actual
# data that training is performed with, and ensure there is more training data than
# INSTEPS.
# =============================================================================

IN_STEPS = 170
OUT_STEPS = 60


#Import data from Yahoo Finance
df = si.get_data(ticker,start_date,end_date,index_as_date,interval)
date_time = pd.to_datetime(df.index, format='%d-%m-%Y %H:%M:%S')

#Prune features and introduce time signals.
df = df.drop(columns = ['open','high','low','close','ticker'])
#df = df.drop(columns = ['ticker'])

# =============================================================================
# timestamp_s = date_time.map(datetime.datetime.timestamp)
# year = (365.2425)*24*60*60
# df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
# df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
# =============================================================================

indicator_bb = ta.volatility.BollingerBands(close=df["adjclose"], window=30, window_dev=2)
df['BB_HIGH'] = indicator_bb.bollinger_hband()-df['adjclose']
df['BB_LOW']= df['adjclose'] - indicator_bb.bollinger_lband()

MACD = ta.trend.MACD(close = df['adjclose'], window_fast=12, window_slow=26, window_sign=9)
df['MACD'] = MACD.macd_diff()

# =============================================================================
# 
# EMA = ta.trend.EMAIndicator(close = df['adjclose'],window=30)
# df['EMA'] = EMA.ema_indicator()
# =============================================================================

#Plot the data
plot_cols = ['adjclose'] #'BB_HIGH','MACD']
plot_features = df[plot_cols].tail(IN_STEPS)
plot_features.index = date_time[-IN_STEPS:]
_ = plot_features.plot(subplots=True)   

print(df.describe().transpose())

# Plotting feature correlation to identify any strong correlations in data.
plt.figure(figsize = (12, 8))
plt.suptitle("FEATURE CORRELATION")
cor = df.corr()
sns.heatmap(cor,annot=True,cmap=plt.cm.Reds)
plt.show()

#Splitting data into training, validation, and testing sets.
column_dict = {name : i for i, name in enumerate(df.columns)}
n = len(df)
train_df  = df[0:int(n*0.6)]
val_df = df[int(n*0.6):int(n)]
test_df = df[int(n*0.6):]

num_features = df.shape[1]

#Normalize the training data
training_mean = train_df.mean()
training_std = train_df.std()

test_mean = test_df.mean()
test_std = test_df.std()

train_df = (train_df - training_mean) / training_std
val_df = (val_df - training_mean) / training_std
test_df = (test_df - training_mean) / training_std


# =============================================================================
# train_df = (train_df - training_mean) / training_std
# val_df = (val_df - val_df.mean()) / val_df.std()
# test_df = (test_df - test_df.mean()) / test_df.std()
# =============================================================================

#Visualize the distribution of features
df_std = (df - training_mean) / training_std
df_std = df_std.select_dtypes([np.number])
df_std = df_std.melt(var_name = "Features", value_name = "Normalized")
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x = "Features", y = "Normalized", data = df_std)


# Initialize window for training

WINDOW = IN_STEPS + OUT_STEPS
multi_window = WindowGenerator(input_width = IN_STEPS,
                               label_width = OUT_STEPS,
                               shift = OUT_STEPS,
                               train_df = train_df,
                               val_df = val_df,
                               test_df = test_df)

train_perf = {}
multi_val_performance = {}
multi_performance = {}

#CNN MODEL
CONV_WIDTH = 10
multi_conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])
history_cnn = compile_and_fit(multi_conv_model, multi_window, patience = 15)


#LINEAR MODEL
multi_linear_model = tf.keras.Sequential([
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])
history_linear = compile_and_fit(multi_linear_model, multi_window, patience = 100)


#DENSE MODEL
multi_dense_model = tf.keras.Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(512, activation='relu'),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])
history_dense = compile_and_fit(multi_dense_model, multi_window, patience = 15)

IPython.display.clear_output()

train_perf['Linear'] = history_linear.history['mean_absolute_error'][-1]
multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_linear_model)

train_perf['Dense'] = history_dense.history['mean_absolute_error'][-1]
multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
multi_performance['Dense'] = multi_dense_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_dense_model)

train_perf['CNN'] = history_cnn.history['mean_absolute_error'][-1]
multi_val_performance['CNN'] = multi_conv_model.evaluate(multi_window.val)
multi_performance['CNN'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_conv_model)


x = np.arange(len(multi_performance))
width = 0.3


plt.figure(figsize = (12, 8))
metric_name = 'mean_absolute_error'
metric_index = multi_conv_model.metrics_names.index('mean_absolute_error')

train_mae = [v for v in train_perf.values()]
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]

plt.bar(x - (width), train_mae, width, label = 'Training')
plt.bar(x , val_mae, width, label='Validation')
#plt.bar(x + width, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(),
           rotation=45)
plt.ylabel(f'MAE (average over all times and outputs)')
_ = plt.legend()
plt.show()



print(np.shape(test_df[-IN_STEPS:]))
data = np.array(test_df[-IN_STEPS:],dtype=np.float32)
ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data = data,
        targets = None,
        sequence_length = IN_STEPS,
        sequence_stride = 1,
        shuffle = True,
        batch_size = 32)

# Conv.
predictions = multi_conv_model.predict(ds, verbose = 0)
j = df.columns.get_loc('adjclose')
count_predictions = tf.reshape(predictions[0,:,j],[1,OUT_STEPS]).numpy()*training_std[j]+training_mean[j]


plt.figure(figsize = (12, 8))
plt.plot(test_df['adjclose'][-IN_STEPS:].apply(lambda x: x*training_std[j] + training_mean[j]), label='Inputs', marker='.', zorder= -10)
prediction_indices = pd.date_range(start= test_df.index[-1] + pd.DateOffset(1), periods = OUT_STEPS, freq = 'B')
plt.scatter(prediction_indices, count_predictions,
                  marker='X', edgecolors='k', label='CNN Predicitions',
                  c='red', s=64)


#Dense
predictions_dense = multi_dense_model.predict(ds, verbose = 0)
count_predictions_dense = tf.reshape(predictions_dense[0,:,j],[1,OUT_STEPS]).numpy()*training_std[j]+training_mean[j]

plt.scatter(prediction_indices, count_predictions_dense,
                  marker='P', edgecolors='k', label='Dense NN Predictions',
                  c='green', s=64)

# Linear
predictions_linear = multi_linear_model.predict(ds, verbose = 0)
count_predictions_linear = tf.reshape(predictions_linear[0,:,j],[1,OUT_STEPS]).numpy()*training_std[j]+training_mean[j]

plt.scatter(prediction_indices, count_predictions_linear,
                  marker='D', edgecolors='k', label='Linear NN Predictions',
                  c='orange', s=64)

plt.legend()
plt.xlabel('Date')

#Plot the training and validation loss for each epoch.
plt.figure(figsize = (12, 8))
plt.plot(history_dense.history['mean_absolute_error'], label='MAE (training data)')
plt.plot(history_dense.history['val_mean_absolute_error'], label='MAE (validation data)')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

#Plot the training and validation loss for each epoch.
plt.figure(figsize = (12, 8))
plt.plot(history_linear.history['mean_absolute_error'], label='MAE (training data)')
plt.plot(history_linear.history['val_mean_absolute_error'], label='MAE (validation data)')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

#Plot the training and validation loss for each epoch.
plt.figure(figsize = (12, 8))
plt.plot(history_cnn.history['mean_absolute_error'], label='MAE (training data)')
plt.plot(history_cnn.history['val_mean_absolute_error'], label='MAE (validation data)')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()





