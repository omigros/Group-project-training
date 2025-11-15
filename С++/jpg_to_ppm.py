from PIL import Image

input_jpg = "test_noisy.jpg"
output_ppm = "test.ppm"

img = Image.open(input_jpg).convert("RGB")
w, h = img.size
pixels = list(img.getdata())

with open(output_ppm, "w") as f:
    f.write("P3\n")
    f.write(f"{w} {h}\n255\n")
    for y in range(h):
        row = pixels[y * w:(y + 1) * w]
        for r, g, b in row:
            f.write(f"{r} {g} {b} ")
        f.write("\n")

print("PPM P3 файл создан:", output_ppm)
