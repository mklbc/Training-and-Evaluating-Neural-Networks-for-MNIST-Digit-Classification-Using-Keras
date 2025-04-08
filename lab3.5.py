import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical

# Verileri yükle
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Verileri ön işleme
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Hedef etiketleri one-hot encoding ile dönüştür
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10) # Test etiketlerine de one-hot encoding uygulayın

# Model oluştur
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Modeli derle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğit
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Modeli test et
score = model.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Tahminleri görselleştir
predictions = model.predict(X_test)

# Tek bir pencerede görselleştir
fig, axes = plt.subplots(3, 5, figsize=(15, 10))
for i in range(15):
  row = i // 5
  col = i % 5
  axes[row, col].imshow(X_test[i], cmap='coolwarm')
  axes[row, col].set_title('Pred: {} Acc: {:.2f}%'.format(np.argmax(predictions[i]), 100 * np.max(predictions[i])), fontsize=8, color='black')
  axes[row, col].set_xticks([])
  axes[row, col].set_yticks([])
  axes[row, col].set_frame_on(True)  # Çerçeve ekle
  plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Kenar boşluklarını ayarla

fig.suptitle('MNIST Tahminleri', fontsize=16, color='black')
plt.tight_layout()
plt.show()
