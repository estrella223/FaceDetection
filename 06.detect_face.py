import cv2, numpy as np

def preprocessing(no):    ## 검출 전처리 수행(검출기가 학습데이터를 잘 검출 할  수 있도록)
    image = cv2.imread('images/face/%02d.jpg' %no, cv2.IMREAD_COLOR)    # 원본 사진 no 불러옴
    if image is None: return None, None    # 이미지가 없으면 반환 x
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 명암도 영상 변환 (특징을 쉽게 잡기 위해)
    gray = cv2.equalizeHist(gray)  # 히스토그램 평활화(한쪽으로 치우친 명암 분포를 균등하게하여 영상의 화질 개선)
    return image, gray    # 원본 영상, 명암도 영상 반환

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")    # 정면얼굴 검출기 로드(학습데이터를 검출하는)
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")  # 눈 검출기 로드(학습데이터를 검출하는)
image, gray = preprocessing(34)  # 34번 영상 파일을 읽어서 전처리 수행
if image is None: raise Exception("영상 파일 읽기 에러")    # 이미지가 없을 때 프린트값

faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100))    # 얼굴 검출
# detectMultiScale(image, scale factor(디폴트 값이 1.1),minNeighbors (이웃 후보사각형 개수),Minisize 검출할 객체 최소크기,maxsize 가능한 객체 최대 크기)
# scale factor는 사각형을 조금씩 키워주면서 연산. 숫자를 키우면 속도는 빨라지지만 특정 크기 얼굴을 검출 못할 가능성이 있음
# 이웃 후보 사각형 개수는. 얼굴 인식할 때 사각형이 하나만 쓰이는게 아니라 좌우위아래 1~2필섹 정도 떨어진 곳도 마킹하게 되는데
# 여기서는 근방에 사각형이 2개 이상 되어야지 얼굴로 판단한다는 뜻/ 기본은 3개 (값이 높아 지면 품질이 높아진다)


if faces.any():    # 얼굴이 검출된다면(값중에 하나라도 가져올 수 있다면...............?)
    x, y, w, h = faces[0]    # 얼굴의 x, y좌표/ w(너비), h(높이) 값 가져옴
    face_image = image[y:y + h, x:x + w]  # 얼굴 영역 영상 가져오기
    eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25, 20))  # 눈 검출 수행
    if len(eyes) == 2:  # 눈 사각형이 검출되면/ 눈 2개가 검출된다면
        for ex, ey, ew, eh in eyes:    # 눈과 관련된것은 ex, ey, ew, eh라고 이름 붙임
            center = (x + ex + ew // 2, y + ey + eh // 2)     # 눈 중심 좌표
            cv2.circle(image, center, 10, (0, 255, 0), 2)  # 눈 중심에 원 그리기 10 = 반지름 , 색, 선의 두께
    else:
        print("눈 미검출")

    cv2.rectangle(image, faces[0], (255, 0, 0), 2)  # 얼굴 검출 사각형 그리기
    cv2.imshow("image", image)

else: print("얼굴 미검출")
cv2.waitKey(0)