# credit: https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/

from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras import backend
from matplotlib import pyplot

# Regression metrics
# prepare sequence
X = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
y = array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# create model
model = Sequential()
model.add(Dense(2, input_dim=1))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity'])

# train model
history = model.fit(X, X, epochs=500, batch_size=len(X), verbose=2)

# plot metrics
#pyplot.plot(history.history['mean_squared_error'])
#pyplot.plot(history.history['mean_absolute_error'])
#pyplot.plot(history.history['mean_absolute_percentage_error'])
#pyplot.plot(history.history['cosine_proximity'])
#pyplot.show()


#Classification metrics
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# train model
history = model.fit(X, y, epochs=400, batch_size=len(X), verbose=2)
# plot metrics
pyplot.plot(history.history['accuracy'])
pyplot.show()


# Custom metrics
# recommend using the backend math functions wherever possible for consistency and execution speed.
def mean_squared_error(y_true, y_pred):
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


model.compile(loss='mse', optimizer='adam', metrics=[rmse])
# train model
history = model.fit(X, X, epochs=500, batch_size=len(X), verbose=2)
# plot metrics
pyplot.plot(history.history['rmse'])
pyplot.show()