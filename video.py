#pip3 install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

# Header Files
import torch 
# import matplotlib
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from extract import number_plate_extract
from ocr import ocr
import warnings
warnings.filterwarnings("ignore")


model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/nehal/Music/Thesis/1st Review Final/yolo_v5 on trained model/best.pt', force_reload=True)

cap=cv2.VideoCapture('test/test2.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)

result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

while True:
    
    _,img=cap.read()
    
    if img is None:
        break
    else:
        height,width,_=img.shape


    results = model(img)
    # results.print()
    # print(results.pandas().xyxy)
    # print(results.pandas().xyxy[0])
    # print(results.pandas().xyxy[1])

    
    # %matplotlib inline 
    img=np.squeeze(results.render())

    # print(results.render())

    cv2.imshow('Image',img)
    result.write(img)
    key=cv2.waitKey(1)

    if key==27:
        break
    elif key==32:
        cv2.imwrite('image.jpg', img)
        # print(results.pandas().xyxy[0])

        x_min=results.pandas().xyxy[0].loc['xmin']
        y_min=results.pandas().xyxy[0].loc['ymin']
        x_max=results.pandas().xyxy[0].loc['xmax']
        y_max=results.pandas().xyxy[0].loc['ymax']

        number_plate_extract(x_min,y_min,x_max,y_max)

        ocr()

        break

cap.release()
result.release()
cv2.destroyAllWindows()



