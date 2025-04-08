# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.optimizers import Adam

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess data
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

# Create the neural network model
model = Sequential()

# Flatten the 28x28 input images into 784-dimensional vectors
model.add(Flatten(input_shape=(28, 28)))

# Add a hidden layer with 64 neurons and ReLU activation function
model.add(Dense(64, activation='relu'))

# Add an output layer with 10 neurons and softmax activation function for classification
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Train the model on the training data
history = model.fit(train_images, train_labels, epochs=5, batch_size=32, validation_data=(test_images, test_labels))

# Generate Training and Test Loss/Accuracy Plots
# Access training history using history.history dictionary
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Visualize Example Predictions
random_indices = np.random.randint(0, len(test_images), 10)
selected_images = test_images[random_indices]
selected_labels = test_labels[random_indices]

predictions = model.predict(selected_images)

for i, (image, label, pred) in enumerate(zip(selected_images, selected_labels, predictions)):
    plt.subplot(5, 2, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}, Prediction: {np.argmax(pred)}')
    plt.axis('off')
plt.show()

# Evaluate the model's performance on the test data
score = model.evaluate(test_images, test_labels)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
