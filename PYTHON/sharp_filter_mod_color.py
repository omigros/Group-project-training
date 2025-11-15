import cv2
import numpy as np

IMAGE_PATH = "test_noisy.jpg"
OUT_PATH = "filtered_result_mod_color.jpg"

# ------------------------------
# Параметры
# ------------------------------
H_DENOISE = 28                 # сила подавления шума
TEMPLATE_WINDOW_SIZE = 9
SEARCH_WINDOW_SIZE = 27

BIL_D = 9
BIL_SIGMA_COLOR = 40
BIL_SIGMA_SPACE = 40

UNSHARP_AMOUNT = 1.0
MASK_RADIUS = 3

LOCAL_MEAN_WINDOW = 31
VAR_WINDOW = 9

# ------------------------------
# Загрузка и перевод в LAB
# ------------------------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Файл '{IMAGE_PATH}' не найден.")

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
L, A, B = cv2.split(lab)

# ------------------------------
# ШАГ 1. Сильное шумоподавление (Non-local Means) для L
# ------------------------------
denoised_L = cv2.fastNlMeansDenoising(L.astype(np.uint8), None,
                                      h=H_DENOISE,
                                      templateWindowSize=TEMPLATE_WINDOW_SIZE,
                                      searchWindowSize=SEARCH_WINDOW_SIZE).astype(np.float32)

# Для A и B каналов используем мягкое шумоподавление
denoised_A = cv2.fastNlMeansDenoising(A.astype(np.uint8), None,
                                      h=H_DENOISE//2,
                                      templateWindowSize=TEMPLATE_WINDOW_SIZE,
                                      searchWindowSize=SEARCH_WINDOW_SIZE).astype(np.float32)

denoised_B = cv2.fastNlMeansDenoising(B.astype(np.uint8), None,
                                      h=H_DENOISE//2,
                                      templateWindowSize=TEMPLATE_WINDOW_SIZE,
                                      searchWindowSize=SEARCH_WINDOW_SIZE).astype(np.float32)

# ------------------------------
# ШАГ 2. Bilateral smoothing (мягкое сохранение краёв)
# ------------------------------
den_bil_L = cv2.bilateralFilter(denoised_L.astype(np.uint8),
                                BIL_D, BIL_SIGMA_COLOR, BIL_SIGMA_SPACE).astype(np.float32)

den_bil_A = cv2.bilateralFilter(denoised_A.astype(np.uint8),
                                BIL_D, BIL_SIGMA_COLOR, BIL_SIGMA_SPACE).astype(np.float32)

den_bil_B = cv2.bilateralFilter(denoised_B.astype(np.uint8),
                                BIL_D, BIL_SIGMA_COLOR, BIL_SIGMA_SPACE).astype(np.float32)

# ------------------------------
# ШАГ 3. Адаптивная карта контуров для L
# ------------------------------
gx = cv2.Sobel(den_bil_L, cv2.CV_32F, 1, 0, 3)
gy = cv2.Sobel(den_bil_L, cv2.CV_32F, 0, 1, 3)
grad_mag = cv2.magnitude(gx, gy)
grad_norm = cv2.normalize(grad_mag, None, 0, 1, cv2.NORM_MINMAX)

edge_mask = cv2.GaussianBlur(grad_norm, (5,5), 0)
edge_mask = np.clip(edge_mask, 0.1, 1.0)

# ------------------------------
# ШАГ 4. Unsharp Mask (контуры) на L
# ------------------------------
blur = cv2.GaussianBlur(den_bil_L, (MASK_RADIUS*2+1, MASK_RADIUS*2+1), 0)
detail = den_bil_L - blur
enhanced_L = den_bil_L + UNSHARP_AMOUNT * detail * edge_mask

# ------------------------------
# ШАГ 5. Компенсация яркости
# ------------------------------
local_mean = cv2.blur(enhanced_L, (LOCAL_MEAN_WINDOW, LOCAL_MEAN_WINDOW))
enhanced_L = enhanced_L - (local_mean - L)

# ------------------------------
# ШАГ 6. CLAHE на L
# ------------------------------
enhanced_L_u8 = cv2.normalize(enhanced_L, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
final_L = clahe.apply(enhanced_L_u8)

# ------------------------------
# ШАГ 7. Сборка LAB и конвертация в BGR
# ------------------------------
final_lab = cv2.merge([final_L,
                       den_bil_A.astype(np.uint8),
                       den_bil_B.astype(np.uint8)])

final_bgr = cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)

cv2.imwrite(OUT_PATH, final_bgr)
print("✅ Сохранено:", OUT_PATH)
