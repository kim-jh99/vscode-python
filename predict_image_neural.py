
import numpy as np
import cv2
from common.image import img2gray, resize, img_show, find_bounding_rects, visualize_rects
from neuralnet_mnist import init_network, predict


network = init_network("sample_weight.pkl")

img = cv2.imread('dataset/27.png')
gray_img = img2gray(img) # 이미지를 흑백으로 변환
rects = find_bounding_rects(gray_img, min_size=(10, 10)) # 이미지에서 숫자 영역(contour)들을 찾아내기

img = visualize_rects(img, rects) # 숫자 영역에 사각형 그리기

for rect in rects: # 각각의 숫자 영역에서 숫자를 추출해서 예측 결과를 출력
    x, y, w, h = rect
    rect_img = gray_img[y:y+h, x:x+w]  # 숫자 부분 잘라내기
    resized_img = resize(rect_img, dsize=(28, 28)).reshape(784) 
    # 숫자 부분을 원하는 크기로 조정하고 필요한 형태로 변환
    # nerualnet 모듈에서 사용하는 형태로 변환 (784,)

    prediction = np.argmax(predict(network, resized_img / 255.0)) # 예측 결과를 출력
    cv2.putText(img, str(prediction), (x, y),
                cv2.FONT_HERSHEY_PLAIN, 3, color=(255, 0, 0), thickness=2)

img_show(img) # 결과 이미지 출력
