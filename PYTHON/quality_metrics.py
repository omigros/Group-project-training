import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt

# === Метрики ===
def psnr(original, processed):
    """Отношение сигнал/шум (ОСШ, PSNR)"""
    mse = np.mean((original.astype(np.float32) - processed.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))

def tvl_estimate(image):
    """Оценка разрешающей способности (ТВЛ) через градиенты"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(sobelx**2 + sobely**2)
    threshold = np.mean(grad) * 2
    edges = np.sum(grad > threshold)
    h, w = gray.shape
    return (edges / (h * w)) * 1000

# === Анализ изображений ===
def evaluate_images(base_dir, original_name, processed_map):
    """Вычисление PSNR и ТВЛ + построение графика"""
    original_path = os.path.join(base_dir, original_name)
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"Не найдено исходное изображение: {original_path}")

    labels = list(processed_map.keys())
    files = list(processed_map.values())

    psnr_vals, tvl_vals = [], []

    print("📊 Результаты оценки качества\n")
    for label, filename in processed_map.items():
        path = os.path.join(base_dir, filename)
        processed = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if processed is None:
            print(f"[!] Файл {filename} не найден.")
            psnr_vals.append(np.nan)
            tvl_vals.append(np.nan)
            continue

        psnr_val = psnr(original, processed)
        tvl_val = tvl_estimate(processed)
        psnr_vals.append(psnr_val)
        tvl_vals.append(tvl_val)

        print(f"{label} ({filename})")
        print(f"  ОСШ (PSNR): {psnr_val:.2f} дБ")
        print(f"  ТВЛ (оценка): {tvl_val:.2f}")
        print("-" * 40)

    # === Графики ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(labels))

    # ---- PSNR ----
    axes[0].bar(x, psnr_vals, color='#4a90e2', edgecolor='black', alpha=0.85)
    axes[0].set_title('ОСШ (PSNR)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('дБ', fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5, axis='y')
    axes[0].set_ylim(0, max(psnr_vals) * 1.25)
    for i, val in enumerate(psnr_vals):
        axes[0].text(i, val + 0.5, f"{val:.2f}", ha='center', fontsize=9)

    # ---- TVL ----
    axes[1].bar(x, tvl_vals, color='#e94e3b', edgecolor='black', alpha=0.85)
    axes[1].set_title('Разрешающая способность (ТВЛ)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('отн. ед.', fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5, axis='y')
    axes[1].set_ylim(0, max(tvl_vals) * 1.25)
    for i, val in enumerate(tvl_vals):
        axes[1].text(i, val + (max(tvl_vals) * 0.02), f"{val:.2f}", ha='center', fontsize=9)

    plt.suptitle('Сравнение фильтров по объективным метрикам качества', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # === Сохранение ===
    save_path = os.path.join(base_dir, "quality_metrics_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n📁 График сохранён как: {save_path}")

    plt.show()

# === Основная программа ===
if __name__ == "__main__":
    base_dir = r"D:\FOLDERS\STUDY\WORK with Kamenskiy\image filter"

    original_name = "test.jpg"
    processed_map = {
        "C++": "filtered_result.jpg",
        "MATLAB": "filtered_result_nopkg2.jpg",
        "Python": "filtered_result_python.jpg"
    }

    evaluate_images(base_dir, original_name, processed_map)
