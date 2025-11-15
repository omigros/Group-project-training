import cv2
import numpy as np

# === Параметры ===
image_path = "test.jpg"        # исходное изображение
output_path = "test_noisy.jpg" # сохранённый результат

# === Настройки интенсивности ===
# Выбери уровень шума: 'low', 'medium', 'high'
noise_level = 'high'

if noise_level == 'low':
    sigma = 15
elif noise_level == 'medium':
    sigma = 35
elif noise_level == 'high':
    sigma = 60
else:
    sigma = 25  # по умолчанию

mean = 0  # среднее значение шума

# === Загрузка ===
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Изображение '{image_path}' не найдено")

# === Добавление шума ===
noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
noisy_img = img.astype(np.float32) + noise
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

# === Сохранение ===
cv2.imwrite(output_path, noisy_img)
print(f"✅ Гауссов шум уровня '{noise_level}' добавлен. Сохранено как {output_path}")
