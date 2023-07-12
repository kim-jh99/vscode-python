import numpy as np
import cv2
from common.image import img2gray, resize, img_show, find_bounding_rects, visualize_rects
from deep_convnet import DeepConvNet


network = DeepConvNet()
network.load_params("params.pkl")


img = cv2.imread('dataset/27.png')
gray_img = img2gray(img)
rects = find_bounding_rects(gray_img, min_size=(10, 10))

# 테두리 영역 보기
img = visualize_rects(img, rects)

for rect in rects:
    x, y, w, h = rect
    rect_img = gray_img[y:y+h, x:x+w]
    resized_img = resize(rect_img, dsize=(28, 28)).reshape(1, 1, 28, 28)
    prediction = np.argmax(network.predict(resized_img / 255.0))
    cv2.putText(img, str(prediction), (x, y),
                cv2.FONT_HERSHEY_PLAIN, 3, color=(255, 0, 0), thickness=2)

img_show(img)
