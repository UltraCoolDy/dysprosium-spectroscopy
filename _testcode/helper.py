import os
import ctypes
os.add_dll_directory(r"C:\Users\dysprosium\anaconda3\envs\py38\Library\bin")

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import numpy as np
from PIL import ImageGrab
import cv2

# Your coordinates from all_monitors.png
EC_IMG = (400, 250, 560, 300)
HL_IMG = (1350, 250, 1540, 300)

# Windows virtual screen origin and size
user32 = ctypes.windll.user32
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

vx = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
vy = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
vw = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
vh = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)

print("Virtual screen origin:", (vx, vy))
print("Virtual screen size  :", (vw, vh))

def img_to_screen(region_img):
    x1, y1, x2, y2 = region_img
    return (x1 + vx, y1 + vy, x2 + vx, y2 + vy)

EC_REGION = img_to_screen(EC_IMG)
HL_REGION = img_to_screen(HL_IMG)

print("EC screen region:", EC_REGION)
print("HL screen region:", HL_REGION)

# Full screenshot of all monitors
full = ImageGrab.grab(all_screens=True)
full_np = np.array(full)

# Draw boxes using image coordinates, not screen coordinates
overlay = full_np.copy()
cv2.rectangle(overlay, (EC_IMG[0], EC_IMG[1]), (EC_IMG[2], EC_IMG[3]), (0, 0, 255), 2)
cv2.rectangle(overlay, (HL_IMG[0], HL_IMG[1]), (HL_IMG[2], HL_IMG[3]), (0, 255, 0), 2)
cv2.imwrite("all_monitors_boxes.png", overlay)

# Grab raw crops using converted screen coords
ec_img = ImageGrab.grab(bbox=EC_REGION, all_screens=True)
hl_img = ImageGrab.grab(bbox=HL_REGION, all_screens=True)

ec_np = np.array(ec_img)
hl_np = np.array(hl_img)

cv2.imwrite("ec_raw.png", ec_np)
cv2.imwrite("hl_raw.png", hl_np)

print("Saved all_monitors_boxes.png, ec_raw.png, hl_raw.png")