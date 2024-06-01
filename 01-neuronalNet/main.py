import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)


# One neuron in one layer
# input_shape = tf.keras.layers.Input(shape=(1,))
# density = tf.keras.layers.Dense(units=1)

# Three neurons in two layers
input_shape = tf.keras.layers.Input(shape=(1,))
hidden1 = tf.keras.layers.Dense(units=3)
hidden2 = tf.keras.layers.Dense(units=3)
output = tf.keras.layers.Dense(units=1)

model = tf.keras.Sequential([input_shape, hidden1, hidden2, output])

model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

print("Training started...")
history = model.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Model trained")

plt.xlabel(" Epoch")
plt.ylabel(" Loss measure")
plt.plot(history.history['loss'])

plt.show()

print("Time for a prediction!")
test_celsius = np.array([100.0], dtype=float)
result = model.predict(test_celsius)
print("The result is:" + str(result[0]) + " Fahrenheit!")

print("Internal model variables")
print(hidden1.get_weights())
print(hidden2.get_weights())
print(output.get_weights())
