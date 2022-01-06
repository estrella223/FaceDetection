import cv2, numpy as np

def preprocessing(no):    # 검출 전처리 수행
    image = cv2.imread('face/%2d.jpg' %no, cv2.IMREAD_COLOR)    # 원본 사진 no 불러옴
    if image is None: return None, None    # 이미지가 없으면 반환 x
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 명암도 영상 변환 (특징을 쉽게 잡기 위해)
    gray = cv2.equalizeHist(gray)  # 히스토그램 평활화
    return image, gray    # 원본 영상, 명암도 영항 반환

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")    # 정면얼굴 검출기
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")  # 눈 검출기
image, gray = preprocessing(60)  # 전처리 수행
if image is None: raise Exception("영상 파일 읽기 에러")    # 이미지가 없을 때 프린트값

faces = face_cascade.detectMultiScale(gray, 1.15, 1, 0, (10, 10))    # 얼굴 검출
                  # detectMultiScale(image, 영상크기감소, 이웃 후보사각형 개수, 과거 함수에서 사용하던 flag, 가능한 객체 최소 크기
                  # 영상 크기 감소는 작을수록 많은 사람 골라냄(대신 느림?), 이웃 후보사각형도 작을수록 많이 골라냄
for (x, y, w, h) in faces:    # 얼굴이 검출된다면(값중에 하나라도 가져올 수 있다면...............?)
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)  # 얼굴 검출 사각형 그리기
    face_image = image[y:y + h, x:x + w]  # 얼굴 영역 영상 가져오기
    eyes = eye_cascade.detectMultiScale(face_image, 1.1, 7, 0, (7, 7))  # 눈 검출 수행
    for (ex, ey, ew, eh) in eyes:    # 눈과 관련된것은 ex, ey, ew, eh라고 이름 붙임
        center = (x + ex + ew // 2, y + ey + eh // 2)     # 눈 중심 좌표
        cv2.circle(image, center, 8, (0, 255, 0), 2)  # 눈 중심에 원 그리기 10 = 반지름

cv2.imshow("image", image)
cv2.waitKey(0)