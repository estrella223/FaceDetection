import numpy as np, cv2

def draw_ellipse(image, roi, ratio, color, thickness=cv2.FILLED):   #draw_ellipse 를 통해 타원을 그린다
    x, y, w, h = roi    #roi 영상처리를 수행 할 때 설정하는 영역
    center = (x + w // 2, y + h // 2)                   # 타원 중심
    size = (int(w * ratio), int(h * ratio))             # 타원 크기
    cv2.ellipse(image, center, size, 0, 0, 360, color, thickness)
            #(이미지, 중심 ,축 절반크기(가로,세로) ,각도 ,시작각도 , 끝각도,   색상  , 선두께)

    return image
## 마스크를 활용하면 필요한 영역에 대해서만 히스토그램을 계산 가능
def make_masks(rois, correct_shape):                              # 영역별 마스크 생성
    base_mask = np.full(correct_shape, 255, np.uint8) # 기본마스크 / 넘파이의 full함수(facecenter와 eyecneter의 값인 correct_shape, 255색,np.uint8데이터타입)
    hair_mask = draw_ellipse(base_mask, rois[3], 0.45, 0,  -1)  # 헤어마스크 / ellipse함수(base_mask , ndarray형식의 rois 3번까지의 값 ,ratio,color=0,-1은 채우기)
    lip_mask = draw_ellipse(np.copy(base_mask), rois[2], 0.45, 255)  # 립 마스크 / 값을 바꾸는것이 아니기에 copy를 씀 / 배열 2번까지의 값, ratio,색상

    masks = [hair_mask, hair_mask, lip_mask, ~lip_mask]  # 헤어마스크와 립마스크 그리고 립아닌 마스크값 모두 마스크에 넣어줌
    masks = [mask[y:y+h,x:x+w] for mask,(x,y, w,h) in zip(masks, rois)]


    return masks

def calc_histo(image, rois, masks):
    bsize = (64, 64, 64)  # 히스토그램 계급 개수(x축 간격)
    ranges = (0,256, 0,256, 0,256)                                 # 각 채널 빈도 범위(x축 범위)
    subs = [image[y:y+h, x:x+w] for x, y, w, h in rois]  #관심영역 참조로 영상 생성
    hists = [cv2.calcHist([sub], [0,1,2], mask, bsize, ranges)
             for sub, mask in zip(subs, masks)] #관심영역 영상 히스토그램
    hists = [ h/np.sum(h) for h in hists]           # 히스토그램값 정규화

    sim1 = cv2.compareHist(hists[2], hists[3], cv2.HISTCMP_CORREL) #입술- 얼굴 유사도
    sim2 = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL) #윗 - 귀밑머리 유사도
    return  sim1, sim2
