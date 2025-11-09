from PIL import Image
import numpy as np

# === Параметры ===
pgm_file = "filtered_result.pgm"
jpg_file = "filtered_result.jpg"

# === Чтение PGM (P2) ===
with open(pgm_file, "r") as f:
    lines = f.readlines()

# Убираем комментарии и пустые строки
lines = [l for l in lines if not l.startswith("#") and l.strip()]
assert lines[0].strip() == "P2", "Формат файла не P2!"

# Читаем размеры и максимальное значение
w, h = map(int, lines[1].split())
max_val = int(lines[2])
data = " ".join(lines[3:]).split()
pixels = np.array([int(p) for p in data], dtype=np.uint8).reshape((h, w))

# === Сохранение в JPG ===
img = Image.fromarray(pixels, mode="L")
img.save(jpg_file)

print(f"✅ Конвертация завершена: {jpg_file}")
