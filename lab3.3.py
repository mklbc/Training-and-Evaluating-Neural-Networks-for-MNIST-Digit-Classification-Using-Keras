import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
# MNIST verilerini yükleyin
(egitim_goruntuleri, egitim_etiketleri), (test_goruntuleri, test_etiketleri) = keras.datasets.mnist.load_data()

# Verileri normalleştirin
egitim_goruntuleri_normal = egitim_goruntuleri.astype(np.float32) / 255.0

# İlk 25 görüntüyü ve etiketlerini seçin
goruntuler = egitim_goruntuleri_normal[0:25]
etiketler = egitim_etiketleri[0:25]

# Görüntüleri Görüntüle
fig, axes = plt.subplots(5, 5, figsize=(10, 10))

for i, (goruntu, etiket) in enumerate(zip(goruntuler, etiketler)):
    # Görüntüyü düzleştirin
    goruntu_duz = goruntu.flatten()
    
    # Alt grafiğe geçin
    axes[int(i / 5), i % 5].imshow(goruntu, cmap='gray')
    
    # Etiketleri görüntüye ekleyin
    axes[int(i / 5), i % 5].set_title(f"Etiket: {etiket}")
    
    # Eksenleri gizleyin
    axes[int(i / 5), i % 5].axis('off')

# Grafiği Gösterin
plt.suptitle("Normalleştirilmiş MNIST Görüntüleri ve Etiketleri", fontsize=12)
plt.tight_layout()
plt.show()
