# predict_sugar.py
# 수박 사진을 입력하면 당도(Brix)를 예측합니다.
# 파이프라인: 이미지 -> YOLO 몸통 추출 -> 특징 A, B, C 계산 -> 회귀 모델 예측
# (특징 D, E 는 추출은 되지만 예측에는 사용하지 않습니다.)

import sys
import cv2
import joblib
import numpy as np
from ultralytics import YOLO

from body_extractor import extract_watermelon_body
from feature_extractor import calculate_watermelon_features

# ────────────────────────────────────────────────────────────
WEIGHT_PATH = "best.pt"
MODEL_PATH = "sugar_model.pkl"

# feature_extractor 가 돌려주는 dict 의 키 (A, B, C 만 사용)
FEATURE_KEYS = ["특징A (형태비율)", "특징B (선명도)", "특징C (면적 %)"]
# ────────────────────────────────────────────────────────────


def predict_sugar(image_path, vision_model=None, sugar_bundle=None):
    """이미지 경로를 받아 (예측당도, 특징dict) 를 반환합니다. 실패 시 (None, 에러메시지)."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, f"이미지를 찾을 수 없습니다: {image_path}"

    # 모델 로드 (외부에서 안 넘겨주면 여기서 로드)
    if vision_model is None:
        vision_model = YOLO(WEIGHT_PATH)
    if sugar_bundle is None:
        sugar_bundle = joblib.load(MODEL_PATH)

    # 1단계: 수박 몸통 마스크
    solid_body_mask, error = extract_watermelon_body(img_bgr, vision_model, visualize=False)
    if error:
        return None, f"몸통 추출 실패: {error}"

    # 2단계: 특징 계산 (A~E 모두 나오지만 A,B,C 만 사용)
    features = calculate_watermelon_features(img_bgr, solid_body_mask, visualize=False)

    # 3단계: A, B, C 로 회귀 예측
    model = sugar_bundle["model"]
    x = np.array([[features[k] for k in FEATURE_KEYS]], dtype=float)
    brix = float(model.predict(x)[0])

    return brix, features


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test1.jpg"
    print(f"🍉 수박 당도 예측 시작 -> {image_path}\n")

    brix, info = predict_sugar(image_path)

    if brix is None:
        print(f"❌ {info}")
        return

    print("📐 추출된 특징:")
    for k in FEATURE_KEYS:
        print(f"   {k}: {info[k]}")
    print()

    print(f"✅ 예측 당도: {brix:.2f} Brix")


if __name__ == "__main__":
    main()
