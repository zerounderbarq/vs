import time
from PIL import ImageGrab
import pyautogui as pag

while True:
    time.sleep(0.01)
    screen = ImageGrab.grab()
    
    pos = pag.position()
    rgb = screen.getpixel(pos)
    
    if rgb == (75, 219, 106) or rgb == (43, 135, 209):
        pag.click()
        print("Clicked")