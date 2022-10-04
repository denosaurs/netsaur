import tensorflow as tf
import numpy as np
import time


start = time.time()

#tf.get_logger().setLevel('INFO')
#tf.autograph.set_verbosity(3)

x_train, y_train = (tf.constant([[0,0],[0,1],[1,0],[1,1]], "float32"), tf.constant([[0],[1],[1],[0]], "float32"))
XOR_True = [(1, 0), (0, 1)]
XOR_False = [(0, 0), (1, 1)]

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(2,)),
  tf.keras.layers.Dense(3, activation=tf.nn.sigmoid),   # hidden layer
  tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)   # output layer
])

model.compile(
              # optimizer='adam',
              loss='binary_crossentropy',
              #loss='mean_squared_error',   # try this too; treat as regression problem
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=5000, verbose=0)
print("Training took", time.time() - start, "ms")

print("XOR True")
for x in XOR_True:
    print(model.predict(np.array([x])))
print("XOR False")
for x in XOR_False:
    print(model.predict(np.array([x])))
