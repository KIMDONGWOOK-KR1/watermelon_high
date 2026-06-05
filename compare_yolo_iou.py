# compare_yolo_iou.py
# 정답(COCO GT) body 마스크 vs YOLO(best.pt) 추론 body 마스크의 IoU 비교.
#  - GT: coco_body_extractor 로 추출
#  - YOLO: best.pt 추론, class 0(body) 중 최대 인스턴스 선택
#  - 동일 형식으로 비교(원폴리곤 / convex hull 두 가지) + IoU·Dice 통계

import os
import numpy as np
import cv2
from ultralytics import YOLO

from coco_body_extractor import load_coco, build_indexes, get_body_polygon, polygon_to_mask

JSON_PATH = "Watermelon.coco/train/_annotations.coco.json"
IMAGE_DIR = "Watermelon.coco/train"
WEIGHT_PATH = "best.pt"
YOLO_BODY_CLASS = 0          # best.pt names: {0:'body', 1:'stripe'}
USE_CONVEX_HULL = True       # 기존 파이프라인과 동일 조건으로 비교


def iou_dice(gt, pred):
    g = gt > 0
    p = pred > 0
    inter = np.logical_and(g, p).sum()
    union = np.logical_or(g, p).sum()
    iou = inter / union if union > 0 else 0.0
    dice = (2 * inter) / (g.sum() + p.sum()) if (g.sum() + p.sum()) > 0 else 0.0
    return iou, dice


def yolo_body_mask(model, img_bgr, use_convex_hull):
    """YOLO 추론 → body(class 0) 최대 인스턴스의 마스크(uint8). 없으면 None."""
    h, w = img_bgr.shape[:2]
    res = model.predict(source=img_bgr, conf=0.3, verbose=False)[0]
    if res.masks is None:
        return None
    classes = res.boxes.cls.cpu().numpy()
    polys = res.masks.xy
    body_idx = np.where(classes == YOLO_BODY_CLASS)[0]
    if len(body_idx) == 0:
        return None
    areas = [cv2.contourArea(polys[i].astype(np.int32)) if len(polys[i]) >= 3 else 0 for i in body_idx]
    best = body_idx[int(np.argmax(areas))]
    poly = polys[best].astype(np.int32)
    if len(poly) < 3:
        return None
    return polygon_to_mask(poly, h, w, use_convex_hull)


def main():
    print("🔬 GT(COCO) vs YOLO(best.pt) body 마스크 IoU 비교")
    print(f"   조건: {'ConvexHull' if USE_CONVEX_HULL else '원폴리곤'} 마스크 기준\n")

    coco = load_coco(JSON_PATH)
    idx = build_indexes(coco)
    images_by_id, images_by_name, body_by_image, _ = idx

    print("🔄 YOLO 로딩...")
    model = YOLO(WEIGHT_PATH)

    ious, dices = [], []
    gt_none, yolo_none = [], []
    worst = []  # (iou, file)

    total = len(coco["images"])
    for n, im in enumerate(coco["images"], 1):
        fn = im["file_name"]
        h, w = im["height"], im["width"]

        # GT
        poly = get_body_polygon(body_by_image, im["id"])
        if poly is None:
            gt_none.append(fn)
            continue
        gt_mask = polygon_to_mask(poly, h, w, USE_CONVEX_HULL)

        # YOLO
        img = cv2.imread(os.path.join(IMAGE_DIR, fn))
        if img is None:
            continue
        pred_mask = yolo_body_mask(model, img, USE_CONVEX_HULL)
        if pred_mask is None:
            yolo_none.append(fn)
            ious.append(0.0)          # 미검출은 IoU 0 으로 집계
            dices.append(0.0)
            worst.append((0.0, fn))
            continue

        iou, dice = iou_dice(gt_mask, pred_mask)
        ious.append(iou)
        dices.append(dice)
        worst.append((iou, fn))

        if n % 20 == 0:
            print(f"   진행 {n}/{total} ...")

    ious = np.array(ious)
    dices = np.array(dices)
    print("\n📊 결과")
    print(f"   비교 대상: {len(ious)}장 (GT 없음 {len(gt_none)}장 제외)")
    print(f"   YOLO 미검출(IoU=0 처리): {len(yolo_none)}장  {yolo_none if yolo_none else ''}")
    print(f"   평균 IoU : {ious.mean():.4f}")
    print(f"   중앙 IoU : {np.median(ious):.4f}")
    print(f"   평균 Dice: {dices.mean():.4f}")
    print(f"   IoU>=0.90: {(ious >= 0.90).sum()}장  ({(ious>=0.90).mean()*100:.1f}%)")
    print(f"   IoU>=0.80: {(ious >= 0.80).sum()}장  ({(ious>=0.80).mean()*100:.1f}%)")
    print(f"   IoU<0.50 : {(ious < 0.50).sum()}장")

    worst.sort(key=lambda t: t[0])
    print("\n   ⚠️ IoU 최저 5장:")
    for iou, fn in worst[:5]:
        print(f"     {iou:.3f}  {fn}")

    print(f"\n   GT 없음(비교 제외): {gt_none}")


if __name__ == "__main__":
    main()
