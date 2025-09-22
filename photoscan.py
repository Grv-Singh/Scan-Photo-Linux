import cv2
import numpy as np
import os, glob, re
import subprocess
import time

scanner_device = "hpaio:/usb/HP_LaserJet_M1005?serial=KJ2J7X0"
outdir = "/root/Desktop/Scans"
os.makedirs(outdir, exist_ok=True)

tmp_scan = f"/tmp/scan-{int(time.time())}.pnm"

# 1. Scan the photo in color
subprocess.run([
    "scanimage",
    "--device", scanner_device,
    "--mode", "Color",
    "--resolution", "300",
    "--format", "pnm"
], stdout=open(tmp_scan, "wb"), check=True)

# 2. Read color image
img_color = cv2.imread(tmp_scan, cv2.IMREAD_COLOR)
if img_color is None:
    raise ValueError("Failed to load scanned image")
h, w = img_color.shape[:2]

# 3. Convert to grayscale and blur
gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

# 4. Edge detection
edges = cv2.Canny(blur, 50, 150)

# 5. Dilate to close gaps
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
edges = cv2.dilate(edges, kernel, iterations=2)

# 6. Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contours:
    x0, y0, x1, y1 = 0, 0, w, h
else:
    # Select contour with largest area
    c = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(c)
    x0, y0, x1, y1 = x, y, x+cw, y+ch

# 7. Find where photo content ends by analyzing intensity changes
# Create a high contrast version to better detect edges
_, high_contrast = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Calculate the average intensity of each row in the detected photo area
row_intensities = []
for row in range(y0, y1):
    row_avg = np.mean(high_contrast[row, x0:x1])
    row_intensities.append(row_avg)

# Find where the intensity changes significantly (photo ends)
if len(row_intensities) > 10:  # Only if we have enough rows
    # Calculate the gradient of intensities
    gradient = np.gradient(row_intensities)

    # Find the point with the largest negative gradient (biggest drop in intensity)
    max_drop_idx = np.argmin(gradient)

    # Only use this if the drop is significant
    if gradient[max_drop_idx] < -10:
        new_bottom = y0 + max_drop_idx
        if new_bottom < y1 and new_bottom > y0 + (y1 - y0) * 0.8:
            y1 = new_bottom

# 8. Alternative approach: Scan from bottom up to find where content ends
if y1 >= h - 10:
    content_found = False
    for row in range(h-1, y0, -1):
        # Calculate the percentage of "non-white" pixels in this row
        non_white_pixels = np.sum(high_contrast[row, x0:x1] < 200)
        total_pixels = x1 - x0
        non_white_ratio = non_white_pixels / total_pixels

        if non_white_ratio > 0.1:
            y1 = min(row + 5, h)
            content_found = True
            break

    if not content_found:
        y1 = y + ch

# 9. Add a small padding to avoid cutting edges
pad = 5
x0 = max(x0 - pad, 0)
y0 = max(y0 - pad, 0)
x1 = min(x1 + pad, w)
y1 = min(y1 + pad, h)

# 10. Crop original color image
crop = img_color[y0:y1, x0:x1]

# 11. Save sequential color PNG
existing = glob.glob(os.path.join(outdir, "scan-*.png"))
nums = []
for f in existing:
    match = re.search(r"scan-(\d+)\.png", f)
    if match:
        nums.append(int(match.group(1)))

n = max(nums) + 1 if nums else 1
outname = os.path.join(outdir, f"scan-{n:03d}.png")
cv2.imwrite(outname, crop)
print(f"Saved tightly cropped COLOR photo as {outname}")
print(f"Crop coordinates: ({x0}, {y0}) -> ({x1}, {y1})")
print(f"Crop dimensions: {x1-x0}x{y1-y0} (WxH)")

# Cleanup
os.remove(tmp_scan)
