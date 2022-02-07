import easyocr
from easyocr import Reader
import cv2
import matplotlib.pyplot as plt

path='number_plate.jpg'
reader = easyocr.Reader(['en'],gpu=False)
result = reader.readtext(path)

print(result)

