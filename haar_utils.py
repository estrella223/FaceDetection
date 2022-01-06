import numpy as np, cv2

def preprocessing(no):  # 검출 전처리
    image = cv2.imread('images/face/%02d.jpg' %no, cv2.IMREAD_COLOR)
    if image is None: return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 명암도 영상 변환
    gray = cv2.equalizeHist(gray)  # 히스토그램 평활화
    return image, gray

def correct_image(image, face_center, eye_centers):  # 기울어진 얼굴에 대한 보정 수행
    pt0, pt1 = eye_centers  # 좌, 우 눈 중심 좌표
    if pt0[0] > pt1[0]: pt0, pt1 = pt1, pt0         # pt0이 pt1보다 크다면 두 좌표를 바꿔줌(눈의 위치 맞춰줌)

    dx, dy = np.subtract(pt1, pt0).astype(float)                   # 두 좌표간 차분 계산
    angle = cv2.fastAtan2(dy, dx)                   # 차분으로 기울기 계산
    rot_mat = cv2.getRotationMatrix2D(face_center, angle, 1)   #회전 행렬 반환
            # getRotationMatrix2D(center(회전의 중심 좌표(x,y)는 튜플), angle(회전 각도 양수는 반시계방향. 음수는 시계방향), scale(추가적인 확대비율, 확대하고싶은 배수를 써주면 됨))
            # 이건 영상 중앙 기준 회전 방법
    size = image.shape[1::-1]                  # 역순으로 변환
    corr_image = cv2.warpAffine(image, rot_mat, size, cv2.INTER_CUBIC)
                    # warpAffine(src(입력 이미지), M(2x3변환 행렬(rot_mat로 반환한 값 받음), dsize(출력 이미지 사이즈), dst(출력 이미지))
                    # dsize - width=columns, height=rows
                    # INTER_CUBIC은 선명한 이미지는 얻을수 있지만 처리속도가 느림
                    # 이건 영상 좌측 상단 기준 회전

    eye_centers = np.expand_dims(eye_centers, axis=0)             # 차원 증가 (눈좌표 행렬의 차원과 회전 변환행렬의 열수를 맞추기 위해)
    corr_centers = cv2.transform(eye_centers, rot_mat)
    corr_centers = np.squeeze(corr_centers, axis=0)              # 차원 감소(원본 눈 좌표와 같은 차원으로 환원)

    return corr_image, corr_centers                 # 보정 결과 반환


# 11.2.5 입술 영역 및 머리 영역 검출
def define_roi(pt, size):
    return np.ravel([pt, size]).astype(int)  # 2원소 튜플 2개에서 4 원소 튜플 1개(1차원)로 변환

def detect_object(center, face):   ## 머리카락 영역과 입술 영역을 계산하는 함수
    w, h = face[2:4]   # 얼굴 영역 크기(w, h)
    center = np.array(center) # 얼굴 중심좌표 ndarray 객체로 변경/ 사칙 연산 가능해짐
    gap1 = np.multiply((w,h), (0.45, 0.65))     # 얼굴 영역 비율 크기 45%, 65%
    gap2 = np.multiply((w,h), (0.20, 0.1))      # 입술 영역 비율 크기 20%, 10%

    pt1 = center - gap1        # 좌상단 평행이동 - 머리 시작좌표
    pt2 = center + gap1             # 우하단 평행이동 - 머리 종료좌표
    hair = define_roi(pt1, pt2-pt1)       # 전체 머리 영역

    size = np.multiply(hair[2:4], (1, 0.4))   # 머리카락 영역 높이 40%
    hair1 = define_roi(pt1, size)             # 윗머리 영역(x,y,w,h)
    hair2 = define_roi(pt2-size, size)             # 귀밑머리 영역

    lip_center = center + (0, h * 0.3)   # 입술 영역 중심좌표 30%
    lip1 = lip_center - gap2    # 좌상단 평행이동
    lip2 = lip_center + gap2         # 우하단 평행이동
    lip = define_roi(lip1, lip2-lip1)  # 입술 영역

    return [hair1, hair2, lip, hair]      # 각 영역을 리스트 구성 후 반환

