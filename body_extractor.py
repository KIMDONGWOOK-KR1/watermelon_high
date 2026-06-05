# body_extractor.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def extract_watermelon_body(img_bgr, vision_model, visualize=False):
    """
    YOLOv8과 Convex Hull을 사용하여 수박의 본체 마스크를 추출합니다.
    """
    results = vision_model.predict(source=img_bgr, conf=0.3, verbose=False)[0]
    
    if results.masks is None:
        return None, "마스크가 감지되지 않았습니다."
        
    classes = results.boxes.cls.cpu().numpy()
    polygons = results.masks.xy 
    
    body_indices = np.where(classes == 0)[0]
    if len(body_indices) == 0:
        return None, "수박 몸통을 찾을 수 없습니다."

    # 가장 큰 면적의 수박 찾기
    areas = [cv2.contourArea(polygons[i].astype(np.int32)) if len(polygons[i]) >= 3 else 0 for i in body_indices]
    biggest_idx = body_indices[np.argmax(areas)]
    body_polygon = polygons[biggest_idx].astype(np.int32)

    # 빈 도화지에 YOLO 원본 폴리곤 마스크 생성 (Convex Hull 미사용)
    # → Hull 은 꼭지 등 돌기를 직선으로 이어 배경을 삼각형으로 채우는 문제가 있어 사용하지 않는다.
    solid_body_mask = np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8)

    if len(body_polygon) < 3:
        return None, "수박 형태가 너무 작거나 불완전합니다."
    cv2.fillPoly(solid_body_mask, [body_polygon], 255)

    # 모폴로지 오프닝: 꼭지(stem) 같은 얇은 돌기 제거 (몸통 형태는 보존)
    # 커널 크기는 몸통 등가지름에 비례 → 이미지 해상도와 무관하게 동작
    area = cv2.countNonZero(solid_body_mask)
    if area > 0:
        k = max(7, int(np.sqrt(area) * 0.03))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        solid_body_mask = cv2.morphologyEx(solid_body_mask, cv2.MORPH_OPEN, kernel)

    # 오프닝 후 가장 큰 덩어리만 남기기 (떨어져 나간 작은 조각/돌기 제거)
    contours, _ = cv2.findContours(solid_body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "수박 형태가 너무 작거나 불완전합니다."
    biggest = max(contours, key=cv2.contourArea)
    solid_body_mask = np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8)
    cv2.fillPoly(solid_body_mask, [biggest], 255)

    # 🌟 [추가됨] 시각화 기능 (visualize=True 일 때만 작동)
    if visualize:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title('1. Original Image', fontsize=14)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(solid_body_mask, cmap='gray')
        plt.title('2. Solid Body Mask (Convex Hull)', fontsize=14)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    return solid_body_mask, None