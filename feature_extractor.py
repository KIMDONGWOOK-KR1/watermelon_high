# feature_extractor.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_watermelon_features(img_bgr, solid_body_mask, visualize=False):
    """
    몸통 마스크를 기반으로 줄무늬를 추출하고 5대 수학적 특징을 계산합니다.
    """
    # 1. 수박 영역만 잘라내기
    watermelon_extracted = cv2.bitwise_and(img_bgr, img_bgr, mask=solid_body_mask)

    # 2. CLAHE (조명 평탄화)
    lab = cv2.cvtColor(watermelon_extracted, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img_clahe_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    
    # 3. 정밀 HSV 줄무늬 타격 (그물망 미세 확장!)
    hsv = cv2.cvtColor(img_clahe_bgr, cv2.COLOR_BGR2HSV)
    
    # 🌟 [영점 조절 3가지]
    # 1. H(색상): 30 -> 25 (약간 노란빛이 도는 끄트머리 줄무늬 허용)
    # 2. S(채도): 40 -> 30 (색이 조금 탁하거나 바랜 줄무늬 허용)
    # 3. V(밝기): 110 -> 125 (빛을 받아 살짝 밝아진 줄무늬 허용)
    lower_green = np.array([18, 25, 0])
    upper_green = np.array([90, 255, 115])
    
    stripe_mask = cv2.inRange(hsv, lower_green, upper_green)
    stripe_mask = cv2.bitwise_and(stripe_mask, stripe_mask, mask=solid_body_mask)
    
    base_mask = cv2.subtract(solid_body_mask, stripe_mask)

    # 🌟 [추가됨] 시각화 기능 (visualize=True 일 때만 작동)
    if visualize:
        plt.figure(figsize=(18, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(watermelon_extracted, cv2.COLOR_BGR2RGB))
        plt.title('3. Extracted Body', fontsize=14)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(img_clahe_bgr, cv2.COLOR_BGR2RGB))
        plt.title('4. CLAHE (Lighting Eq)', fontsize=14)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(stripe_mask, cmap='gray')
        plt.title('5. Final Stripe Mask', fontsize=14)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    # 4. 5대 특징 수학적 계산
    x, y, w, h = cv2.boundingRect(solid_body_mask)
    feature_A = round(min(w, h) / max(w, h), 3) if max(w, h) > 0 else 0

    v_channel = hsv[:, :, 2]
    base_mean_v = cv2.mean(v_channel, mask=base_mask)[0] if cv2.countNonZero(base_mask) > 0 else 0
    stripe_mean_v = cv2.mean(v_channel, mask=stripe_mask)[0] if cv2.countNonZero(stripe_mask) > 0 else 0
    feature_B = round(base_mean_v - stripe_mean_v, 2)

    body_pixels = cv2.countNonZero(solid_body_mask)
    stripe_pixels = cv2.countNonZero(stripe_mask)
    feature_C = round((stripe_pixels / body_pixels) * 100, 2) if body_pixels > 0 else 0

    # dist_transform = cv2.distanceTransform(stripe_mask, cv2.DIST_L2, 5)
    # feature_D = round(cv2.mean(dist_transform, mask=stripe_mask)[0], 2) if stripe_pixels > 0 else 0

    # contours_stripe, _ = cv2.findContours(stripe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # valid_stripes = [c for c in contours_stripe if cv2.contourArea(c) > 50]
    # feature_E = len(valid_stripes)

    return {
        '특징A (형태비율)': feature_A,
        '특징B (선명도)': feature_B,
        '특징C (면적 %)': feature_C,
        # '특징D (평균굵기)': feature_D,
        # '특징E (파편화/가닥)': feature_E
    }