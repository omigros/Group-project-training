# sharp_filter_modificated.py
import cv2
import numpy as np

# === Параметры ===
BASE_A1 = 2.0       # базовый a1
BASE_A2 = 20.0      # базовый a2 (центр)
GAIN = 0.25         # общий множитель на маску
IMAGE_PATH = 'test_noisy.jpg'  # изображение должно лежать в той же папке

# === Загрузка ===
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Файл '{IMAGE_PATH}' не найден. Помести его в папку со скриптом.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

# === Базовые матрицы ===
mask_base = np.array([
    [-1, -4, -8, -10, -8, -4, -1],
    [-4, -16, -32, -40, -32, -16, -4],
    [-8, -32, 17, 82, 17, -32, -8],
    [-10, -40, 82, 224, 82, -40, -10],
    [-8, -32, 17, 82, 17, -32, -8],
    [-4, -16, -32, -40, -32, -16, -4],
    [-1, -4, -8, -10, -8, -4, -1]
], dtype=np.float32)

small_3x3 = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32)

# === Вспомогательные функции ===
def make_oriented_small_masks(n_dirs=8):
    """Возвращает список 3x3 масок small_3x3, повернутых на углы (в градусах)."""
    masks = []
    center = (1, 1)
    for k in range(n_dirs):
        ang = k * 360.0 / n_dirs
        M = cv2.getRotationMatrix2D(center, ang, 1.0)
        # warpAffine требует float32 image; используем интерполяцию
        rotated = cv2.warpAffine(small_3x3, M, (3, 3), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        masks.append(rotated)
    return masks

def make_full_masks(oriented_smalls, a1_scale=1.0, a2_scale=1.0):
    """Для каждого ориентационного small строим полную 7x7 маску (без DC и без глобального GAIN)."""
    fulls = []
    for small in oriented_smalls:
        add = np.zeros((7,7), dtype=np.float32)
        add[2:5,2:5] = small * (BASE_A1 * a1_scale)
        add[3,3] += BASE_A2 * a2_scale
        m = mask_base + add
        m = m - np.mean(m)   # DC компенсация локально по маске
        m = m * GAIN
        fulls.append(m)
    return fulls

def circular_diff_deg(a, b):
    """Минимальная разность углов в градусах (|a-b| в цикле 360)."""
    d = np.abs(a - b) % 360.0
    d = np.minimum(d, 360.0 - d)
    return d

# === Карта контраста и ориентации ===
# локальная контрастная карта: разница и локальное std/absdiff
blur = cv2.GaussianBlur(gray, (9,9), 0)
contrast_map = cv2.absdiff(gray, blur)
if contrast_map.max() > 0:
    contrast_map = contrast_map / contrast_map.max()
else:
    contrast_map = contrast_map

gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
angle = cv2.phase(gx, gy, angleInDegrees=True)  # 0..360

# === Предвычислим отклики по направлениям ===
N_DIRS = 8
oriented_smalls = make_oriented_small_masks(N_DIRS)

# Для стабильности используем a1,a2 масштаб 1 для всех масок; локальная адаптация выполнится при комбинировании
full_masks = make_full_masks(oriented_smalls, a1_scale=1.0, a2_scale=1.0)

# Выполним свёртки (filter2D) для каждого направления на всем изображении
responses = []
for idx, m in enumerate(full_masks):
    # filter2D требует ядро типа float32; применяем к исходному gray
    resp = cv2.filter2D(gray, cv2.CV_32F, m, borderType=cv2.BORDER_REPLICATE)
    responses.append(resp)
responses = np.stack(responses, axis=0)  # shape (N_DIRS, H, W)

# === Объединение откликов по ориентированности и контрасту ===
# для каждого пикселя вычисляем веса по углу: чем ближе угол к направлению маски, тем больше вес
dir_angles = np.array([k * 360.0 / N_DIRS for k in range(N_DIRS)], dtype=np.float32)  # degrees

# Преобразуем angle в матрицу для сравнения: shape (N_DIRS, H, W)
H, W = gray.shape
angle_rep = np.broadcast_to(angle[np.newaxis, :, :], (N_DIRS, H, W))

# Разница по углам и веса (гаусс)
sigma_deg = 45.0  # ширина окна ориентационной привязки (можно настраивать)
ang_diff = circular_diff_deg(angle_rep, dir_angles[:, None, None])
weights_angle = np.exp(-(ang_diff**2) / (2 * sigma_deg**2))

# Контрастная модуляция: усиливаем отклик там, где контраст высок
# создаём map scale = 1 + k * contrast (k задаёт насколько адаптивно усиливаем)
k_contrast = 1.2
contrast_scale = 1.0 + k_contrast * contrast_map  # shape (H,W)
contrast_rep = np.broadcast_to(contrast_scale[np.newaxis,:,:], (N_DIRS, H, W))

# Искомый комбинированный отклик R = sum_dir ( w_dir(ang) * resp_dir * contrast_scale )
weighted = responses * weights_angle * contrast_rep
combined = np.sum(weighted, axis=0) / (np.sum(weights_angle * contrast_rep, axis=0) + 1e-6)

# === Каскады: добавляем комбинированный отклик в несколько проходов с разным коэффициентом ===
result = gray.copy()
scales = [1.0, 0.7, 0.5, 0.3]  # 4 каскада
for s in scales:
    result = result + combined * s

# === Локальная нормализация/DC-компенсация ===
# Сохраним среднюю яркость на локальных окнах: вычтем локальное среднее, затем добавим глобальную (или нормализуем)
local_mean = cv2.blur(result, (15,15))
result = result - (local_mean - local_mean.mean())

# Нормализуем в 0..255
res_norm = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

# === Сохранение и вывод короткой информации ===
OUT_NAME = "filtered_result_modificated.jpg"
cv2.imwrite(OUT_NAME, res_norm)
print("Фильтрация завершена. Результат сохранён как:", OUT_NAME)

# Если хочешь видеть также карту контрастов и направления как файлы:
cv2.imwrite("contrast_map.png", (contrast_map * 255).astype(np.uint8))
# направление (угол/360 -> 0..255)
cv2.imwrite("orientation_map.png", (angle / 360.0 * 255.0).astype(np.uint8))
