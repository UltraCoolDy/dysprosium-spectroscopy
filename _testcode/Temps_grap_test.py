import os
import ctypes
os.add_dll_directory(r"C:\Users\dysprosium\anaconda3\envs\py38\Library\bin")

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import numpy as np
from PIL import ImageGrab
import cv2

# Coordinates taken from all_monitors.png
EC_IMG = (400, 250, 560, 300)
HL_IMG = (1350, 250, 1540, 300)

user32 = ctypes.windll.user32
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77

VX = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
VY = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)

def img_to_screen(region_img):
    x1, y1, x2, y2 = region_img
    return (x1 + VX, y1 + VY, x2 + VX, y2 + VY)

EC_REGION = img_to_screen(EC_IMG)
HL_REGION = img_to_screen(HL_IMG)

def read_temp(region):
    img = ImageGrab.grab(bbox=region, all_screens=True)
    img = np.array(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = r'--psm 7 -c tessedit_char_whitelist=0123456789.'
    text = pytesseract.image_to_string(thresh, config=config)

    cleaned = "".join(c for c in text if c.isdigit() or c == ".").strip()

    if not cleaned:
        raise RuntimeError("OCR failed to read temperature")

    return int(round(float(cleaned)))

def get_oven_temps():
    ec = read_temp(EC_REGION)
    hl = read_temp(HL_REGION)
    return ec, hl

if __name__ == "__main__":
    ec, hl = get_oven_temps()
    print(f"EC: {ec}")
    print(f"HL: {hl}")