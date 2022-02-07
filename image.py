#pip3 install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

# Header Files
import torch 
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np
import cv2
import os

model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/nehal/Music/Thesis/1st Review Final/yolo_v5 on trained model/best.pt', force_reload=True)

img = 'C:\\Users\\nehal\\Music\\Thesis\\1st Review Final\\yolo_v5 on trained model\\3f796298e3bd8b8b.jpg'

results = model(img)

print('Result:  \n')
results.print()
df=results.pandas().xyxy[0]
print(df.columns)
print(df['xmin'])
# print(type(results.pandas().xyxy[0]))
# print(len(results.pandas().xyxy[0]))
# print(results.pandas().xyxy[0].loc['x_min'])


#%matplotlib inline 
img=np.squeeze(results.render())

cv2.imshow('Image',img)
key=cv2.waitKey(1)



cv2.destroyAllWindows()
