import tensorflow as tf
from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt
mnist=keras.dataset.mnist
x_train,x_test=x_train/255.0,x_test/255.0
model=keras.model.Sequential([
    keras.layers.flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
]

)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

# Make a prediction on the first test image
predictions = model.predict(x_test)
predicted_label = np.argmax(predictions[0])

# Display the image and predicted label
plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.title(f"Predicted: {predicted_label}")
plt.show()
