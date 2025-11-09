from PIL import Image

# Загружаем исходное JPG
img = Image.open("test.jpg").convert("L")  # L = grayscale

# Сохраняем в формате PGM (ASCII)
with open("test.pgm", "w") as f:
    w, h = img.size
    pixels = list(img.getdata())
    f.write("P2\n")
    f.write(f"{w} {h}\n255\n")
    for y in range(h):
        row = pixels[y * w:(y + 1) * w]
        f.write(" ".join(str(p) for p in row) + "\n")

print("✅ test.pgm успешно создан (в формате P2)")
