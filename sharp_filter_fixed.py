import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Параметры фильтра ===
a1 = 2      # коэффициент подъема апертуры 3x3
a2 = 20     # коэффициент подъема центрального элемента
gain = 0.25  # усиление фильтра (как в C++)

image_path = 'test.jpg'  # путь к изображению

# === Загрузка изображения ===
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError("Изображение не найдено. Помести test.jpg рядом со скриптом")

# Перевод в оттенки серого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

# === Маска 7x7 (основная) ===
mask_base = np.array([
    [-1, -4, -8, -10, -8, -4, -1],
    [-4, -16, -32, -40, -32, -16, -4],
    [-8, -32, 17, 82, 17, -32, -8],
    [-10, -40, 82, 224, 82, -40, -10],
    [-8, -32, 17, 82, 17, -32, -8],
    [-4, -16, -32, -40, -32, -16, -4],
    [-1, -4, -8, -10, -8, -4, -1]
], dtype=np.float32)

# === Маска апертуры 3x3 ===
mask_3x3 = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32)

# === Добавляем 3x3 в центр 7x7 с коэффициентами a1 и a2 ===
mask_add = np.zeros((7, 7), dtype=np.float32)
mask_add[2:5, 2:5] = mask_3x3 * a1
mask_add[3, 3] += a2

# === Итоговая маска ===
mask_final = mask_base + mask_add
mask_final = (mask_final - np.mean(mask_final)) * gain

# === Применяем фильтр ===
filtered = cv2.filter2D(gray, -1, mask_final)

# === Добавляем контуры к исходному изображению (как в MATLAB) ===
sharp = gray + filtered

# === Нормализация (0–255) ===
sharp = cv2.normalize(sharp, None, 0, 255, cv2.NORM_MINMAX)
sharp = sharp.astype(np.uint8)
gray = gray.astype(np.uint8)

# === Сохраняем и показываем ===
cv2.imwrite("filtered_result_python.jpg", sharp)
print("✅ Фильтрация завершена. Результат: filtered_result_python.jpg")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('До фильтра')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sharp, cmap='gray')
plt.title(f'После фильтра (a1={a1}, a2={a2}, gain={gain})')
plt.axis('off')

plt.tight_layout()
plt.show()
