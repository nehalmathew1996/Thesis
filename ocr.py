import easyocr
from easyocr import Reader
import cv2
import matplotlib.pyplot as plt
# import numpy as np

def ocr():
    path='number_plate.jpg'
    reader = easyocr.Reader(['en'],gpu=False)
    result = reader.readtext(path)

    print(result)
    return 0

    # top_left = tuple(result[0][0][0])
    # bottom_right = tuple(result[0][0][2])
    # text = result[0][1]
    # font = cv2.FONT_HERSHEY_SIMPLEX


    # img = cv2.imread(path)
    # img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),5)
    # img = cv2.putText(img,text,top_left,font,.5,(255,255,255),2,cv2.LINE_AA)

    # cv2.imwrite('ocr.jpg', img)


