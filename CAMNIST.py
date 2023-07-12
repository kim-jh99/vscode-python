import cv2
import numpy as np
from common.image import find_bounding_rects, img2gray, resize
from deep_convnet import DeepConvNet
import pyttsx3
import time

# 웹캠 초기화
cap = cv2.VideoCapture(0)                               # OpenCV 라이브러리를 사용하여 웹캠을 초기화 0은 노트북 내장캠

# MNIST 신경망 모델 로드
network = DeepConvNet()                                 # DeepCNN 사용을 위한 인스턴스 생성
network.load_params("params.pkl")                       # 미리 학습된 params.pkl을 load

# 웹캠 프레임 크기 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)                  # 웹캠의 가로 세로 크기를 설정한다 현재 640X480
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 인식 주기 및 시간 변수 초기화
recognition_interval = 1  # 인식 주기 (초)               # 숫자 인식에 딜레이를 주어 너무 빠르게 연속적으로 인식되지 않도록 설정
last_recognition_time = time.time()                     # 일정 시간이 경과한 경우에만 숫자 인식을 수행

# pyttsx3를 이용하여 음성 출력
def speak(text):
    engine = pyttsx3.init()                             # 함수 호출 후 음성인식 초기화
    engine.setProperty('rate', 150)                     # 음성 속도 설정 (기본값: 200)
    engine.setProperty('volume', 1)                     # 음성 볼륨 설정 (기본값: 1.0)
    engine.say(text)                                    # 출력할 테스트 설정
    engine.runAndWait()                                 # 음성 출력 실행

while True:                                             # 무한루프 실행
    ret, frame = cap.read()                             # 웹캠 프레임 읽기

    current_time = time.time()                          # 현재 시간 저장
    elapsed_time = current_time - last_recognition_time # 현재 시간과 마지막 인식 시간과의 차이 계산 후 시간 확인

    if cv2.waitKey(1) == ord(' '):                      # 스페이스바를 눌렀을 때 실행
        if elapsed_time >= recognition_interval:
            # 프레임 전처리 과정
            gray_img = img2gray(frame)
            rects = find_bounding_rects(gray_img, min_size=(50, 50)) # 50, 50 사이즈의 숫자가 있는 경계상자를 찾음

            for rect in rects:                          # 찾은 경계상자 순회
                x, y, w, h = rect    
                rect_img = gray_img[y:y+h, x:x+w]       # 이미지 추출

                resized_img = resize(rect_img, dsize=(28, 28)).reshape(1, 1, 28, 28) # 이미지를 28X28 크기로 조정하고 모델에 입력할 형식으로 변환
                prediction = np.argmax(network.predict(resized_img / 255.0))         # 모델을 사용하여 숫자 예측

                if prediction in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:                     # 예측된 숫자가 0~9 까지 속하는지 확인
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)         # 경계 상자 그리기

                    cv2.putText(frame, str(prediction), (x, y-10),                   # 숫자 출력
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

                    speak(str(prediction))                                           # 숫자 음성 출력

            last_recognition_time = current_time                                     # 마지막 인식 시간 업데이트

    cv2.imshow('MNIST Webcam', frame)                                                # 경계 상자 그리기

    # 종료 조건
    if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty('MNIST Webcam', cv2.WND_PROP_VISIBLE) < 1:
         break

# 종료
cap.release()
cv2.destroyAllWindows()
