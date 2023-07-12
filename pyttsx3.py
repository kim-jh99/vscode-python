import cv2
import numpy as np
from common.image import img2gray, resize, find_bounding_rects
from deep_convnet import DeepConvNet
import pyttsx3

# MNIST 신경망 모델 로드
network = DeepConvNet()
network.load_params("params.pkl")

# 웹캠 초기화
cap = cv2.VideoCapture(0)

# 웹캠 프레임 크기 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# pyttsx3를 이용하여 음성 출력하는 함수
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # 음성 속도 설정 (기본값: 200)
    engine.setProperty('volume', 0.8)  # 음성 볼륨 설정 (기본값: 1.0)
    engine.say(text)
    engine.runAndWait()

while True:
    # 웹캠 프레임 읽기
    ret, frame = cap.read()

    # 프레임 전처리
    gray_img = img2gray(frame)
    rects = find_bounding_rects(gray_img, min_size=(150, 150))

    # 경계 상자 및 숫자 인식
    recognized_rects = []
    for rect in rects:
        x, y, w, h = rect
        rect_img = gray_img[y:y+h, x:x+w]
        resized_img = resize(rect_img, dsize=(28, 28)).reshape(1, 1, 28, 28)
        prediction = np.argmax(network.predict(resized_img / 255.0))

        if prediction in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            recognized_rects.append(rect)
            # 경계 상자 그리기
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # 숫자 출력
            cv2.putText(frame, str(prediction), (x, y-10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            # 숫자 음성 출력
            speak(str(prediction))

    # 화면에 출력
    cv2.imshow('MNIST Webcam', frame)

    if len(recognized_rects) == 0:
        cv2.waitKey(5000)
    else:
        cv2.waitKey(5000)  # 인식된 경우 일시 정지

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()
