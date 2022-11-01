import requests
import cv2
import numpy as np
import exp_testing as exp
import os
import torch
from PIL import Image
url = "http://192.168.0.19:8080/shot.jpg"
print(os.getcwd())


def predict(im):
  
  im = Image.fromarray(im.astype(np.uint8))
  im1=exp.image_loader(im)
  cnn = exp.CNNModel()
  ckpt = torch.load('C:/Users/LAXMI NARSIMHA/Desktop/Video_Captrue_Testing/exp_classifier.t7',map_location='cpu')
  cnn.load_state_dict(ckpt['net_dict'])
  cnn.eval()
  outputs = cnn(im1)

  prediction = torch.max(outputs.data, 1)[1]
  return prediction.item()


while True:
    img = requests.get(url)
    arr = np.array(bytearray(img.content),dtype=np.uint8)
    im = cv2.imdecode(arr,-1)
    image_frame = im
    #cv2.imshow("AndroidCam",im)
    prediction = predict(im)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    image = cv2.putText(image_frame, exp.classes[prediction],org,font,fontScale ,color,thickness, cv2.LINE_AA )
    cv2.imshow('frame', image_frame)

    if cv2.waitKey(1) == 27:
        break
