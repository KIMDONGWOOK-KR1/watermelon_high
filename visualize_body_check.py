# visualize_body_check.py
# 새 best.pt 가 수박 body 를 잘 뽑는지 "눈으로" 확인하기 위한 오버레이 이미지를 저장합니다.
# 각 샘플마다 [원본 | 마스크 | 오버레이] 를 가로로 붙여 body_check/ 에 저장.
#
# 사용법:
#   python visualize_body_check.py            # 데이터셋에서 12장 샘플
#   python visualize_body_check.py img1.jpg img2.jpg ...   # 지정한 파일만

import os
import sys
import glob

import cv2
import numpy as np
from ultralytics import YOLO

from body_extractor import extract_watermelon_body

WEIGHT_PATH = "best.pt"
IMG_DIR = "Watermelon.coco/train"
OUT_DIR = "body_check"
N_SAMPLE = 12


def make_panel(img_bgr, mask):
    """[원본 | 마스크 | 오버레이] 가로 결합 이미지 반환."""
    h, w = img_bgr.shape[:2]
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 오버레이: 마스크 영역을 초록으로 반투명 + 외곽선
    overlay = img_bgr.copy()
    green = np.zeros_like(img_bgr); green[:] = (0, 200, 0)
    sel = mask > 0
    overlay[sel] = cv2.addWeighted(img_bgr, 0.5, green, 0.5, 0)[sel]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), max(2, w // 300))

    panel = np.hstack([img_bgr, mask_bgr, overlay])
    # 너무 크면 가로 1500px 로 축소
    if panel.shape[1] > 1500:
        scale = 1500 / panel.shape[1]
        panel = cv2.resize(panel, (1500, int(panel.shape[0] * scale)))
    return panel


def pick_files(argv):
    if len(argv) > 1:
        return argv[1:]
    files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    if not files:
        return []
    # 고르게 N_SAMPLE 장 샘플링
    step = max(1, len(files) // N_SAMPLE)
    return files[::step][:N_SAMPLE]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = pick_files(sys.argv)
    if not files:
        print("❌ 대상 이미지를 찾지 못했습니다.")
        return

    print(f"🔄 모델 로딩: {WEIGHT_PATH}")
    model = YOLO(WEIGHT_PATH)
    print(f"🖼️  {len(files)}장 처리 → {OUT_DIR}/\n")

    ok, fail = 0, 0
    for f in files:
        name = os.path.basename(f)
        # IMG_DIR 안 파일명만 준 경우 경로 보정
        path = f if os.path.exists(f) else os.path.join(IMG_DIR, f)
        img = cv2.imread(path)
        if img is None:
            print(f"   ⚠️ 읽기 실패: {name}")
            fail += 1
            continue
        mask, err = extract_watermelon_body(img, model, visualize=False)
        if err:
            print(f"   ❌ 추출 실패: {name}  ({err})")
            fail += 1
            continue
        panel = make_panel(img, mask)
        out = os.path.join(OUT_DIR, f"check_{os.path.splitext(name)[0]}.jpg")
        cv2.imwrite(out, panel)
        cover = cv2.countNonZero(mask) / (img.shape[0] * img.shape[1]) * 100
        print(f"   ✅ {name}  (body 면적 {cover:.1f}%) → {os.path.basename(out)}")
        ok += 1

    print(f"\n완료: 성공 {ok} · 실패 {fail}   결과 폴더: {OUT_DIR}/")
    print("   각 이미지는 [원본 | 마스크 | 오버레이] 순서입니다.")


if __name__ == "__main__":
    main()
