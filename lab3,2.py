import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

(egitim_goruntuleri, egitim_etiketleri), (test_goruntuleri, test_etiketleri) = keras.datasets.mnist.load_data()

# İlk görüntüyü yeniden şekillendirin (Çözüm 1)
ilk_goruntu = egitim_goruntuleri[0].reshape(28, 28)

# VEYA (Çözüm 2)
# ilk_goruntu = egitim_goruntuleri[0].reshape(784)

# Görüntüyü çizin
plt.imshow(ilk_goruntu, cmap='gray')  # 2 boyutlu dizi için imshow kullanın (Çözüm 1)
# plt.matshow(ilk_goruntu, cmap='gray')  # 1 boyutlu dizi için matshow kullanın (Çözüm 2)
plt.show()
