import requests
import cv2
import numpy as np

url = "http://192.168.0.19:8080/shot.jpg"

while True:
    img = requests.get(url)
    arr = np.array(bytearray(img.content),dtype=np.uint8)
    im = cv2.imdecode(arr,-1)

    cv2.imshow("AndroidCam",im)

    if cv2.waitKey(1) == 27:
        break
