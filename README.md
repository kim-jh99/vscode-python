## 개발 환경

python 3.8 이상

```
numpy==1.24.2
matplotlib==3.6.3
urllib3==1.26.14
opencv-python==4.7.0.68
pyttsx3==2.90
```
## 주요 파일 설명

```
├── params.pkl                        # 학습후 생성되는 weights 파일(Accuracy : 0.9935)
├── testcam                           # mnist 숫자 인식 테스트 (캠 사용)
├── train_convnet.py                  # mnist 학습(Accuracy 향상)
└── CAMNIST                           # mnist 웹캠을 통한 음성출력 프로그램(완성본)
```

## 명령어

#### 학습하기

```bash
$ python train_convnet.py
```

#### 테스트하기

```bash
$ python testcam                      # mnist 숫자 인식 테스트 (캠 사용)
```

#### 실행하기

```bash
$ python CAMNIST                      # mnist 웹캠을 사용한 숫자인식 음성출력 시스템 (캠 사용)
```

### 구체 설명

```bash
Step 1. CAMNIST 파일을 실행한다.
Step 2. 손글씨로 출력을 원하는 숫자를 적는다.
Step 3. A4용지에 적은 손글씨 숫자를 웹캠에 정확히 비춘다. (숫자 외 공간은 흰 바탕이여야 정확도가 더욱 상승한다.)
Step 4. Spacebar를 누르면 프로그램이 숫자를 인식하여 음성으로 출력한다.
EX 여러개의 숫자도 한번에 인식이 가능 하며 예를들어 주민등록번호를 인식한다고 하면 인식 가능함.

