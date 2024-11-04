import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
              [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
              [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
              [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]])
y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_dim=4, activation="tanh"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x, y, epochs=100)
data = model.fit(x, y, epochs=100, batch_size=4)

loss, accuracy = model.evaluate(x, y)
print("loss", loss)
print("accuracy", accuracy)

plt.subplot(211)
plt.title('Loss')
plt.plot(data.history['loss'])
plt.subplot(212)
plt.title('Accuracy')
plt.plot(data.history['accuracy'])
plt.show()

prediction = model.predict(x)
for inp, pred in zip(x, prediction):
    print(inp, round(pred[0]))