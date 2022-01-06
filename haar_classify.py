##분류 및 영상 출력 함수
# 유사도를 비교하여 입술-얼굴 유사도와, 윗-귀밑머리 유사도를 비교하여 성별 분류 하는 함수

import cv2
from haar_histogram import draw_ellipse
from Common_utils import put_string

def classify(image, sims, no):
    criteria = 0.25 if sims[0] > 0.2 else 0.1            # 얼굴-입술 유사도
    value = sims[1] > criteria                           #윗머리-귀밑머리 유사도

    text = "Woman" if value else "Man"
    text = '%02dg: ' %no + text                          #분류 결과 출력
    result = "유사도 [입술-얼굴: %4.3f 윗-귀밑머리: %4.3f]" % (sims)
    put_string(image, text, (10, 30), "")           # 영상 출력
    print(text + " - " + result)                    # 콘솔창 출력
def display(image, face_center, centers, sub):

    face_center = int(face_center[0]), int(face_center[1])
    cv2.circle(image, face_center, 3, (0, 0, 255), 2)	    # 얼굴 중심점 표시

    cv2.circle(image, tuple(centers[0]), 10, (0, 255, 0), 2)	# 눈 표시

    cv2.circle(image, tuple(centers[1]), 10, (0, 255, 0), 2)

    draw_ellipse(image, sub[2], 0.35,(0, 0, 255),  2)	    # 얼굴 타원
    draw_ellipse(image, sub[3], 0.45,(255, 100, 0), 2)     # 입술 타원
    cv2.imshow("correct_image", image)

