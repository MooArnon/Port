import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

data = pd.read_csv('/Users/moomacprom1/Data_science/Code/GitHub/Tutorials/Machine_Learning/ConcreteComponent_CompressiveStrength/Concrete_Data.csv')

#* Checking data
"""
print(data.head())
print(data.shape)
print(data.describe())
print(data.isna().any())
"""

#* Data preprocessing
data_train = data.sample(frac=0.8, random_state=0)
data_test = data.drop(data_train.index)
train_label = data_train['ConcreteCompressiveStrength']
test_label = data_test['ConcreteCompressiveStrength']
features = data.drop(['ConcreteCompressiveStrength'], axis=1).columns

print(features)
def modelBuild():
    model = keras.Sequential([
        keras.layers.Dense(16, activation=tf.nn.relu ,input_dim=9),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(1)        
    ])
    optimizer = keras.optimizers.RMSprop(0.0001)
    # Compile
    model.compile(loss='mse',
                  optimizer = optimizer,
                  metrics=['mae','mse'])
    return model

model = modelBuild()
earlyStop = keras.callbacks.EarlyStopping(monitor='mse', patience=250)
history = model.fit(data_train, train_label, 
                    epochs=7500, validation_split=0.2, 
                    callbacks=earlyStop)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch


#* Measure how good of this model
def plot_model_mse():
    plt.xlabel('Epochs')
    plt.ylabel('Mean Square Error')
    plt.title('Mean Square Errors')
    plt.legend()
    plt.plot(hist['epoch'], hist['mse'])
    plt.plot(hist['epoch'], hist['val_mse'])
    plt.show()
def plot_model_mae():
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Errors')
    plt.legend()
    plt.plot(hist['epoch'], hist['mae'])
    plt.plot(hist['epoch'], hist['val_mae'])
    plt.show()


print(data_test)
#* Prediction
test_prediction = model.predict(data_test)
# Plotting
plot_model_mse()
plot_model_mae()

plt.scatter(test_label, test_prediction)
plt.xlabel('True Values')
plt.ylabel('Predictions')

plt.scatter(test_label, test_prediction)
plt.xlabel('True Values')
plt.ylabel('Predictions')
_ = plt.plot([-100,100], [-100, 100])
plt.show()

model.summary()
