import cv2

# 웹캠 초기화
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 의미합니다. 다른 번호로 변경하여 다른 웹캠을 사용할 수도 있습니다.

while True:
    # 웹캠 프레임 읽기
    ret, frame = cap.read()

    # 웹캠 영상 출력
    cv2.imshow('WebCam', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
         break

# 웹캠 종료
cap.release()
cv2.destroyAllWindows()
