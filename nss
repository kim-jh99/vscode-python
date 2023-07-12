import cv2
import numpy as np
from common.image import img2gray, resize, find_bounding_rects
from deep_convnet import DeepConvNet
from AppKit import NSSpeechSynthesizer

# MNIST 신경망 모델 로드
network = DeepConvNet()
network.load_params("params.pkl")

# 웹캠 초기화
cap = cv2.VideoCapture(0)

# 웹캠 프레임 크기 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# NSSpeechSynthesizer 객체 생성
speech = NSSpeechSynthesizer.alloc().initWithVoice_("com.apple.speech.synthesis.voice.samantha")

is_running = True

while is_running:
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
            speech.startSpeakingString_(str(prediction))

    # 화면에 출력
    cv2.imshow('MNIST Webcam', frame)

    if len(recognized_rects) == 0:
        cv2.waitKey(100)
    else:
        cv2.waitKey(500)  # 인식된 경우 일시 정지

    # 종료 조건 체크
    if cv2.getWindowProperty('MNIST Webcam', cv2.WND_PROP_VISIBLE) < 1:
        is_running = False

# 종료
cap.release()
cv2.destroyAllWindows()
