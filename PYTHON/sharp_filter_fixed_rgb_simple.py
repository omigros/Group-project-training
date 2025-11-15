import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Параметры фильтра ===
a1 = 2      # коэффициент подъема апертуры 3x3
a2 = 20     # коэффициент подъема центрального элемента
gain = 0.25  # усиление фильтра

IMAGE_PATH = 'test_noisy.jpg'
OUT_PATH = 'filtered_result_fixed_rgb_simple.jpg'

# === Загрузка изображения ===
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Изображение '{IMAGE_PATH}' не найдено.")

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

# === Применяем фильтр к каждому каналу B, G, R ===
channels = cv2.split(img.astype(np.float32))
filtered_channels = []
for ch in channels:
    filtered = cv2.filter2D(ch, -1, mask_final)
    sharp_ch = ch + filtered
    sharp_ch = np.clip(sharp_ch, 0, 255)  # ограничиваем значения
    filtered_channels.append(sharp_ch.astype(np.uint8))

# === Собираем обратно в BGR ===
sharp_rgb = cv2.merge(filtered_channels)

# === Сохраняем результат ===
cv2.imwrite(OUT_PATH, sharp_rgb)
print("✅ Фильтрация завершена. Результат:", OUT_PATH)

# === Отображение ===
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('До фильтра')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(sharp_rgb, cv2.COLOR_BGR2RGB))
plt.title(f'После фильтра (a1={a1}, a2={a2}, gain={gain})')
plt.axis('off')

plt.tight_layout()
plt.show()
