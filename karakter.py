import cv2
import numpy as np
import os

# Buat folder output
os.makedirs("output", exist_ok=True)

# 1. Membuat gambar "burung"

# Kanvas putih
canvas = np.full((500, 400, 3), 255, dtype=np.uint8)

# Badan burung (oval biru)
cv2.ellipse(canvas, (200, 260), (100, 50), 0, 0, 360, (255, 0, 0), -1)

# Kepala burung (biru muda)
cv2.circle(canvas, (280, 230), 40, (255, 150, 100), -1)

# Paruh (segitiga oranye)
pts = np.array([[300, 230], [280, 227], [280, 240]], np.int32)
cv2.fillPoly(canvas, [pts], (0, 165, 255))

# Sayap (biru tua)
cv2.ellipse(canvas, (190, 260), (60, 30), 330, 0, 360, (200, 0, 0), -1)

# Ekor (3 garis)
cv2.line(canvas, (100, 250), (60, 240), (100, 0, 0), 10)
cv2.line(canvas, (100, 260), (60, 260), (100, 0, 0), 10)
cv2.line(canvas, (100, 270), (60, 280), (100, 0, 0), 10)

# Mata (putih + hitam)
cv2.circle(canvas, (295, 220), 10, (255, 255, 255), -1)
cv2.circle(canvas, (295, 220), 5, (0, 0, 0), -1)
cv2.circle(canvas, (270, 220), 10, (255, 255, 255), -1)
cv2.circle(canvas, (270, 220), 5, (0, 0, 0), -1)

# Kaki (kuning)
cv2.line(canvas, (190, 300), (180, 320), (0, 255, 255), 6)
cv2.line(canvas, (210, 300), (220, 320), (0, 255, 255), 6)

# Simpan gambar burung
cv2.imwrite("output/burung.png", canvas)


# 2. Transformasi

M_trans = np.float32([[1, 0, 50], [0, 1, 30]])
translated = cv2.warpAffine(canvas, M_trans, (400, 500))
cv2.imwrite("output/translate.png", translated)

rows, cols = canvas.shape[:2]
M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), 25, 1)
rotated = cv2.warpAffine(canvas, M_rot, (cols, rows))
cv2.imwrite("output/rotate.png", rotated)

resized = cv2.resize(canvas, None, fx=0.6, fy=0.6)
cv2.imwrite("output/resize.png", resized)

cropped = canvas[150:350, 100:300]
cv2.imwrite("output/crop.png", cropped)


# 3. Operasi Bitwise

bg = np.full((500, 400, 3), (0, 0, 0), dtype=np.uint8)

red = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(red, 155, 250, cv2.THRESH_BINARY_INV)
mask_inv = cv2.bitwise_not(mask)

fg = cv2.bitwise_and(canvas, canvas, mask=mask)
bg_part = cv2.bitwise_and(bg, bg, mask=mask_inv)

bitwise_result = cv2.add(bg_part, fg)
cv2.imwrite("output/bitwise.png", bitwise_result)


# 4. Gambar Akhir (Final)
final = bitwise_result.copy()
cv2.putText(final, "Hewan Burung", (100, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,80), 2)
cv2.imwrite("output/final.png", final)


# 5. Tampilkan hasil
cv2.imshow("Burung Asli", canvas)
cv2.imshow("Translasi", translated)
cv2.imshow("Rotasi", rotated)
cv2.imshow("Resize", resized)
cv2.imshow("Crop", cropped)
cv2.imshow("Bitwise", bitwise_result)
cv2.imshow("Final", final)

cv2.waitKey(0)
cv2.destroyAllWindows()
