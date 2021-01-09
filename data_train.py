from stock_prediction import load_data
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

ticker = "AAPL"
start_date = "01/01/2018"
end_date = "01/01/2021"
index_as_date = True
interval = "1d"

#Import data from Yahoo Finance
df = load_data(ticker,start_date,end_date,index_as_date,interval)
date_time = pd.to_datetime(df.index, format='%d-%m-%Y %H:%M:%S')

#Plot the data
plot_cols = ['adjclose', 'volume']
plot_features = df[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)

#Prune features and introduce time signals.
df = df.drop(columns = ['open','high','low','close','ticker'])

timestamp_s = date_time.map(datetime.datetime.timestamp)
year = (365.2425)*24*60*60
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

print(df.describe().transpose())

#Splitting data into training, validation, and testing sets.
column_dict = {name : i for i, name in enumerate(df.columns)}
n = len(df)
train_df  = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

#Normalize the training data
training_mean = train_df.mean()
training_std = train_df.std()

train_df = (train_df - training_mean) / training_std
val_df = (val_df - training_mean) / training_std
test_df = (test_df - training_mean) / training_std

#Visualize the distribution of features
df_std = (df - training_mean) / training_std
df_std = df_std.select_dtypes([np.number])
df_std = df_std.melt(var_name = "Features", value_name = "Normalized")
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x = "Features", y = "Normalized", data = df_std)

  
# Initialize window for training
OUT_STEPS = 30
multi_window = WindowGenerator(input_width = 60,
                               label_width = OUT_STEPS,
                               shift = OUT_STEPS, train_df = train_df, val_df = val_df, test_df = test_df)

multi_val_performance = {}
multi_performance = {}

CONV_WIDTH = 3
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

history = compile_and_fit(multi_conv_model, multi_window)


IPython.display.clear_output()

multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_conv_model)