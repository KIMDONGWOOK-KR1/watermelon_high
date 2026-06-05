# retrain_sugar_model.py
# 리뉴얼한 best.pt(YOLO 몸통추출)로 원본 이미지에서 특징 A,B,C 를 다시 추출한 뒤,
# 당도(Brix) 회귀 모델을 재학습/평가/저장합니다. (특징 D, E 는 사용하지 않습니다.)
#
# 출력물:
#   - 수박_특징_리뉴얼.csv  : 새 모델로 재추출한 특징 기록 (재현용)
#   - sugar_model.pkl       : 새 특징으로 학습한 회귀 모델 (배포용, 기존 파일 덮어씀)

import os

import cv2
import numpy as np
import pandas as pd
import joblib
from ultralytics import YOLO

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

from body_extractor import extract_watermelon_body
from feature_extractor import calculate_watermelon_features

# ────────────────────────────────────────────────────────────
WEIGHT_PATH = "best.pt"
CSV_IN = "최종_수박_학습데이터_원본120.csv"   # 파일명 + 정답 당도(Brix) 출처
CSV_OUT = "수박_특징_리뉴얼.csv"               # 새 모델로 재추출한 특징 기록
MODEL_OUT = "sugar_model.pkl"
IMG_DIR = "Watermelon.coco/train"

FEATURE_KEYS = ["특징A (형태비율)", "특징B (선명도)", "특징C (면적 %)"]
TARGET_COL = "당도 (Brix)"
FILENAME_COL = "파일명"
# ────────────────────────────────────────────────────────────


def extract_features_with_new_model(df):
    """새 best.pt 로 각 이미지에서 특징 A,B,C 를 재추출해 DataFrame 으로 반환."""
    print(f"🔄 새 몸통추출 모델 로딩: {WEIGHT_PATH}")
    model = YOLO(WEIGHT_PATH)

    records, fail = [], []
    total = len(df)
    for _, row in df.iterrows():
        fname = row[FILENAME_COL]
        img = cv2.imread(os.path.join(IMG_DIR, fname))
        if img is None:
            fail.append((fname, "이미지 없음"))
            continue
        mask, err = extract_watermelon_body(img, model, visualize=False)
        if err:
            fail.append((fname, err))
            continue
        f = calculate_watermelon_features(img, mask, visualize=False)
        records.append({
            FILENAME_COL: fname,
            FEATURE_KEYS[0]: f[FEATURE_KEYS[0]],
            FEATURE_KEYS[1]: f[FEATURE_KEYS[1]],
            FEATURE_KEYS[2]: f[FEATURE_KEYS[2]],
            TARGET_COL: row[TARGET_COL],
        })
        if (len(records) + len(fail)) % 20 == 0:
            print(f"   진행 {len(records) + len(fail)}/{total} ...")

    if fail:
        print(f"   ⚠️ 추출 실패 {len(fail)}개: {[f[0] for f in fail[:10]]}")
    return pd.DataFrame.from_records(records)


def evaluate(name, model, X, y, cv):
    y_pred = cross_val_predict(model, X, y, cv=cv)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)
    accuracy = (1 - mape) * 100
    print(f"  [{name:18s}]  MAE={mae:.3f} Brix   RMSE={rmse:.3f}   "
          f"R^2={r2:.3f}   정확도={accuracy:.1f}%")
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "accuracy": accuracy}


def main():
    print("🍉 리뉴얼 모델 기반 수박 당도 회귀 재학습 시작")
    print(f"   사용 특징: {FEATURE_KEYS}  (D, E 미사용)\n")

    src = pd.read_csv(CSV_IN).dropna(subset=[FILENAME_COL, TARGET_COL]).reset_index(drop=True)
    feat_df = extract_features_with_new_model(src)

    # 재추출 특징을 CSV 로 기록 (재현/검증용)
    feat_df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print(f"\n💾 재추출 특징 저장 → {CSV_OUT}  ({len(feat_df)}행)")

    X = feat_df[FEATURE_KEYS].to_numpy(dtype=float)
    y = feat_df[TARGET_COL].to_numpy(dtype=float)
    print(f"   학습 샘플 수: {len(y)}개")
    print(f"   당도 범위: {y.min():.1f} ~ {y.max():.1f} Brix (평균 {y.mean():.2f})\n")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", RandomForestRegressor(n_estimators=300, random_state=42)),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", GradientBoostingRegressor(random_state=42)),
        ]),
    }

    print("📊 교차검증(5-Fold) 성능 비교:")
    metrics = {name: evaluate(name, m, X, y, cv) for name, m in candidates.items()}

    best_name = min(metrics, key=lambda n: metrics[n]["mae"])
    best_model = candidates[best_name]
    best_metric = metrics[best_name]
    print(f"\n🏆 최적 모델: {best_name} (MAE={best_metric['mae']:.3f} Brix)")

    best_model.fit(X, y)
    joblib.dump({
        "model": best_model,
        "feature_cols": FEATURE_KEYS,
        "model_name": best_name,
        "cv_mae": best_metric["mae"],
        "cv_rmse": best_metric["rmse"],
        "cv_r2": best_metric["r2"],
        "cv_mape": best_metric["mape"],
        "cv_accuracy": best_metric["accuracy"],
        "n_samples": int(len(y)),
        "feature_source": "best.pt (리뉴얼) 재추출",
    }, MODEL_OUT)
    print(f"💾 회귀 모델 저장 완료 → {MODEL_OUT}")


if __name__ == "__main__":
    main()
