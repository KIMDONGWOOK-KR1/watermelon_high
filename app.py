# app.py
# Gradio 웹 인터페이스: 수박 사진을 업로드하면 당도(Brix)를 예측합니다.
# 파이프라인: 이미지 -> YOLO 몸통 추출 -> 특징 A, B, C 계산 -> 회귀 모델 예측
# (특징 D, E 는 표시만 하고 예측에는 사용하지 않습니다.)

import cv2
import joblib
import numpy as np
import gradio as gr
from ultralytics import YOLO

from body_extractor import extract_watermelon_body
from feature_extractor import calculate_watermelon_features

# ────────────────────────────────────────────────────────────
WEIGHT_PATH = "best.pt"
MODEL_PATH = "sugar_model.pkl"
FEATURE_KEYS = ["특징A (형태비율)", "특징B (선명도)", "특징C (면적 %)"]
# ────────────────────────────────────────────────────────────

# 모델은 앱 시작 시 한 번만 로드 (매 예측마다 다시 안 읽음)
print("🔄 모델 로딩 중...")
VISION_MODEL = YOLO(WEIGHT_PATH)
SUGAR_BUNDLE = joblib.load(MODEL_PATH)
SUGAR_MODEL = SUGAR_BUNDLE["model"]
CV_MAE = SUGAR_BUNDLE.get("cv_mae")          # 교차검증 평균오차 (Brix)
CV_RMSE = SUGAR_BUNDLE.get("cv_rmse")        # 교차검증 RMSE (Brix)
CV_ACCURACY = SUGAR_BUNDLE.get("cv_accuracy")  # 정확도 % (100 - MAPE%)
CV_R2 = SUGAR_BUNDLE.get("cv_r2")            # 결정계수 (설명력)
print(f"✅ 준비 완료 (회귀 모델: {SUGAR_BUNDLE.get('model_name', '?')})")


def predict(image_rgb):
    """Gradio 입력(RGB numpy) -> (예측당도 텍스트, 특징 표 dict)."""
    if image_rgb is None:
        return "이미지를 업로드해 주세요.", {}

    # Gradio 는 RGB 로 주므로 파이프라인용 BGR 로 변환
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # 1단계: 수박 몸통 마스크
    solid_body_mask, error = extract_watermelon_body(img_bgr, VISION_MODEL, visualize=False)
    if error:
        return f"❌ 몸통 추출 실패: {error}", {}

    # 2단계: 특징 계산 (A~E 추출, 예측엔 A,B,C 만 사용)
    features = calculate_watermelon_features(img_bgr, solid_body_mask, visualize=False)

    # 3단계: A, B, C 로 회귀 예측
    x = np.array([[features[k] for k in FEATURE_KEYS]], dtype=float)
    brix = float(SUGAR_MODEL.predict(x)[0])

    # 오차 범위 표시: 교차검증 평균오차(MAE)를 ± 로 보여줌
    if CV_MAE is not None:
        lo, hi = brix - CV_MAE, brix + CV_MAE
        result_text = (
            f"🍉 예측 당도: {brix:.2f} Brix  (± {CV_MAE:.2f})\n"
            f"   예상 범위: {lo:.2f} ~ {hi:.2f} Brix\n"
            f"   ✅ 모델 정확도: 약 {CV_ACCURACY:.1f}%  (설명력 R²: {CV_R2 * 100:.1f}%)\n"
            f"   ※ 평균오차 MAE={CV_MAE:.2f}, RMSE={CV_RMSE:.2f} Brix (5-Fold 교차검증 기준)"
        )
    else:
        result_text = f"🍉 예측 당도: {brix:.2f} Brix"
    return result_text, features


with gr.Blocks(title="수박 당도 예측기") as demo:
    _acc = f"{CV_ACCURACY:.1f}%" if CV_ACCURACY is not None else "?"
    _r2 = f"{CV_R2 * 100:.1f}%" if CV_R2 is not None else "?"
    gr.Markdown(
        "# 🍉 수박 당도 예측기\n"
        "수박 사진을 업로드하면 **특징 A·B·C**를 추출해 당도(Brix)를 예측합니다. "
        "(특징 D·E는 예측에 사용하지 않습니다.)\n\n"
        f"**모델 정확도: 약 {_acc}** · 설명력(R²): {_r2} · "
        f"평균오차: ±{CV_MAE:.2f} Brix"
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="수박 사진 업로드")
            btn = gr.Button("당도 예측하기", variant="primary")
            gr.Examples(
                examples=["test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg"],
                inputs=inp,
                label="예시 이미지",
            )
        with gr.Column():
            out_text = gr.Textbox(label="예측 결과", lines=4)
            out_feats = gr.JSON(label="추출된 특징 (A~E, 예측엔 A·B·C만 사용)")

    btn.click(fn=predict, inputs=inp, outputs=[out_text, out_feats])


if __name__ == "__main__":
    demo.launch()
