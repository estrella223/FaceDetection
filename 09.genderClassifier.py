##

from haar_utils import * # 검출기 적재 및 전처리 함수
from haar_histogram import * # 히스토그램 비교 관련 함수
from haar_classify import *

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")


no, max_no, cnt = 0, 60, 1
while True:
    no = no + cnt
    image, gray = preprocessing(no) # 전처리 수행
    if image is None: # 영상읽기 예외처리
        print("%02d.jpg: 영상파일 없음" % no)
        if no < 0 : no = max_no-1
        elif no >= max_no: no = 0
        continue

    faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100)) # 얼굴 검출
    if faces.any():
        x, y, w, h = faces[0]
        face_image = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25, 20))

        if len(eyes) == 2:
            face_center = (x+w//2.0, y+h//2.0) # 얼굴 영역 영상 가져오기
            eye_centers = [(x+ex+ew//2, y+ey+eh//2) for ex, ey, ew, eh in eyes]
            corr_image, corr_center = correct_image(image, face_center, eye_centers)

            rois = detect_object(face_center, faces[0]) #4개영역 계산
            masks = make_masks(rois, corr_image.shape[:2]) # 각 영역 마스크 생성
            sims = calc_histo(corr_image, rois, masks) #히스토그램 생성

            classify(corr_image, sims, no) # 성별 분류 및 표시
            display(corr_image, face_center, corr_center, rois)  #얼굴, 눈 표시


        else:
            print("%02d.jpg: 눈 미검출" % no)

    else:
        print("%02d.jpg: 얼굴 미검출" % no)

    key = cv2.waitKeyEx(0)  #키 이벤트 대기
    if key == 2490368: cnt = 1 # 위쪽 화살표 키이면 다음 영상
    elif key == 2621440: cnt = -1 # 아래쪽 화살표 키이면 이전 영상
    elif key == 32 or key == 27: break # 프로그램 종료 조건
cv2.destroyAllWindows()