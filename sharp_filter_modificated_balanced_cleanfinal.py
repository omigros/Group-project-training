import cv2
import numpy as np

IMAGE_PATH = "test_noisy.jpg"
OUT_PATH = "filtered_result_modificated_balanced_cleanfinal.jpg"

# ------------------------------
# Параметры
# ------------------------------
H_DENOISE = 28                 # сила подавления шума
TEMPLATE_WINDOW_SIZE = 9
SEARCH_WINDOW_SIZE = 27

BIL_D = 9
BIL_SIGMA_COLOR = 40
BIL_SIGMA_SPACE = 40

UNSHARP_AMOUNT = 1.0           # сила контурного усиления
MASK_RADIUS = 3

LOCAL_MEAN_WINDOW = 31
VAR_WINDOW = 9

# ------------------------------
# Загрузка
# ------------------------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Файл '{IMAGE_PATH}' не найден.")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

# ------------------------------
# ШАГ 1. Сильное шумоподавление (Non-local Means)
# ------------------------------
denoised = cv2.fastNlMeansDenoising(gray.astype(np.uint8), None,
                                    h=H_DENOISE,
                                    templateWindowSize=TEMPLATE_WINDOW_SIZE,
                                    searchWindowSize=SEARCH_WINDOW_SIZE).astype(np.float32)

# ------------------------------
# ШАГ 2. Bilateral smoothing (мягкое сохранение краёв)
# ------------------------------
den_bil = cv2.bilateralFilter(denoised.astype(np.uint8),
                              BIL_D, BIL_SIGMA_COLOR, BIL_SIGMA_SPACE).astype(np.float32)

# ------------------------------
# ШАГ 3. Адаптивная карта контуров
# ------------------------------
gx = cv2.Sobel(den_bil, cv2.CV_32F, 1, 0, 3)
gy = cv2.Sobel(den_bil, cv2.CV_32F, 0, 1, 3)
grad_mag = cv2.magnitude(gx, gy)
grad_norm = cv2.normalize(grad_mag, None, 0, 1, cv2.NORM_MINMAX)

# карта усиления (где контуры — усиливаем, где фон — не трогаем)
edge_mask = cv2.GaussianBlur(grad_norm, (5,5), 0)
edge_mask = np.clip(edge_mask, 0.1, 1.0)

# ------------------------------
# ШАГ 4. Unsharp Mask (восстановление чёткости)
# ------------------------------
blur = cv2.GaussianBlur(den_bil, (MASK_RADIUS*2+1, MASK_RADIUS*2+1), 0)
detail = den_bil - blur
enhanced = den_bil + UNSHARP_AMOUNT * detail * edge_mask

# ------------------------------
# ШАГ 5. Компенсация яркости
# ------------------------------
local_mean = cv2.blur(enhanced, (LOCAL_MEAN_WINDOW, LOCAL_MEAN_WINDOW))
enhanced = enhanced - (local_mean - gray)

# ------------------------------
# ШАГ 6. Нормализация и вывод
# ------------------------------
final = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
final = clahe.apply(final)

cv2.imwrite(OUT_PATH, final)
print("✅ Сохранено:", OUT_PATH)
