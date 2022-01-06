from haar_utils import *  #correct_image(기울어진 얼굴에 대한 보정 수행)
                          #머리카락 영역과 입술 영역을 계산하는 함수

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")# 얼굴 검출기
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")# 눈 검출기
image, gray = preprocessing(34) #전처리
if image is None: raise Exception("영상파일 읽기 에러")

faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100)) # 얼굴 검출
# detectMultiScale(image, scale factor(디폴트 값이 1.1),minNeighbors (이웃 후보사각형 개수),Minisize 검출할 객체 최소크기,maxsize 가능한 객체 최대 크기)
if faces.any() :
    x, y, w, h = faces[0]
    face_image = image[y:y+h, x:x+w]  # 얼굴 영역 영상 가져오기
    eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25, 20)) # 눈 검출

    if len(eyes) == 2:
        face_center = (x+w//2.0, y+h//2.0)
        eye_centers = [(x+ex+ew//2, y+ey+eh//2) for ex, ey, ew, eh in eyes]
        corr_image, corr_center = correct_image(image, face_center, eye_centers)# 회전 보정
        rois = detect_object(face_center, faces[0])  # 머리 및 입술영역 계산

        cv2.rectangle(corr_image, rois[0], (255, 0, 255), 2) #윗머리 영역
        cv2.rectangle(corr_image, rois[1], (255, 0, 255), 2) #귓머리 영역
        cv2.rectangle(corr_image, rois[2], (255, 0, 0), 2) #입술 영역
                #rectangle(img ,시작좌표~종료좌표, 색상, 선의 두께)
        cv2.circle(corr_image, tuple(corr_center[0]), 5, (0, 255, 0), 2) #보정 눈 좌표
        cv2.circle(corr_image, tuple(corr_center[1]), 5, (0, 255, 0), 2) # 보정 눈 좌표
        face_center = int(face_center[0]), int(face_center[1])
        cv2.circle(corr_image, face_center, 3, (0, 0, 255), 2) # 얼굴 중심좌표
              #circle(img  , center(원의중심좌표),반지름,색,선의 두께)
        cv2.imshow("correct_image", corr_image)
    else:
        print("눈 미검출")
else:
    print("얼굴 미검출")
cv2.imshow("image", image)
cv2.waitKey(0)


