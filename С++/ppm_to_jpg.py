from PIL import Image
import numpy as np

input_ppm = "filtered.ppm"
output_jpg = "filtered.jpg"

with open(input_ppm, "r") as f:
    lines = f.readlines()

lines = [l.strip() for l in lines if l.strip() and not l.startswith("#")]

assert lines[0] == "P3"

w, h = map(int, lines[1].split())
maxv = int(lines[2])

data = " ".join(lines[3:]).split()
data = list(map(int, data))

arr = np.array(data, dtype=np.uint8).reshape((h, w, 3))

img = Image.fromarray(arr, mode="RGB")
img.save(output_jpg)

print("Конвертация завершена:", output_jpg)
