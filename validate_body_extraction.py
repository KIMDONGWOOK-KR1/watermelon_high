# validate_body_extraction.py
# body_polygon_extraction_design.md 설계 로직이 "가장 body를 잘 뽑는지" 교차검증한다.
#  - 설계 로직(이름으로 카테고리 조회 + 면적최대 선택 + 퇴화필터)을 구현
#  - 121장 전체에 적용해 커버리지/정확성 측정
#  - 대안 전략(첫번째 선택 / 면적필터 없음 / convex hull)과 정량 비교

import json
import os
from collections import defaultdict

import cv2
import numpy as np

JSON_PATH = "Watermelon.coco/train/_annotations.coco.json"
BODY_NAME = "body"


# ──────────────────────────────────────────────────────────
# 설계 로직 구현
# ──────────────────────────────────────────────────────────
def load_coco(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_indexes(coco):
    images_by_id = {im["id"]: im for im in coco["images"]}
    # 카테고리는 '이름'으로 조회 (설계 원칙)
    body_id = next(c["id"] for c in coco["categories"] if c["name"] == BODY_NAME)
    body_by_image = defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] == body_id:
            body_by_image[a["image_id"]].append(a)
    return images_by_id, body_by_image, body_id


def valid_candidates(anns, use_area_filter=True):
    out = []
    for a in anns:
        seg = a.get("segmentation")
        if not seg or isinstance(seg, dict):     # 빈 값 / RLE 제외 (← 31번 1차 방어선)
            continue
        poly = seg[0]
        if len(poly) < 6:                        # 점 3개 미만 제외
            continue
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        true_area = cv2.contourArea(pts)         # COCO 'area'(=bbox면적) 대신 실면적
        if use_area_filter and true_area <= 0:   # 퇴화 폴리곤 제외(안전망)
            continue
        out.append((true_area, poly))
    return out


def pick(anns, strategy="max_area", use_area_filter=True):
    """strategy: 'max_area' | 'first'"""
    cands = valid_candidates(anns, use_area_filter=use_area_filter)
    if not cands:
        return None
    if strategy == "first":
        return np.array(cands[0][1], dtype=np.int32).reshape(-1, 2)
    _, best = max(cands, key=lambda t: t[0])
    return np.array(best, dtype=np.int32).reshape(-1, 2)


def polygon_to_mask(poly, h, w, use_convex_hull=False):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = cv2.convexHull(poly) if use_convex_hull else poly
    cv2.fillPoly(mask, [pts], 255)
    return mask


def poly_bbox(poly):
    x, y, w, h = cv2.boundingRect(poly)
    return [x, y, w, h]


# ──────────────────────────────────────────────────────────
# 교차검증
# ──────────────────────────────────────────────────────────
def main():
    coco = load_coco(JSON_PATH)
    images_by_id, body_by_image, body_id = build_indexes(coco)
    n_img = len(coco["images"])
    print("🍉 Body 폴리곤 추출 로직 교차검증")
    print(f"   이미지 {n_img}장 · body category_id={body_id} (이름 'body'로 조회)\n")

    # ── [검증 1] 설계 로직(max_area + filter) 커버리지 ──────────
    extracted, missing = 0, []
    bbox_ok, area_ratios, hull_inflation = 0, [], []
    for im in coco["images"]:
        iid, fn = im["id"], im["file_name"]
        h, w = im["height"], im["width"]
        anns = body_by_image.get(iid, [])
        poly = pick(anns, "max_area", True)
        if poly is None:
            missing.append(fn)
            continue
        extracted += 1

        # bbox 일치 검증: 선택된 폴리곤의 원본 어노테이션 bbox 와 비교
        sel = max(valid_candidates(anns, True), key=lambda t: t[0])
        sel_poly = sel[1]
        ann_bbox = next(a["bbox"] for a in anns
                        if a.get("segmentation") and a["segmentation"][0] == sel_poly)
        coco_area = next(a["area"] for a in anns
                         if a.get("segmentation") and a["segmentation"][0] == sel_poly)
        pb = poly_bbox(poly)
        if all(abs(pb[k] - ann_bbox[k]) <= 2 for k in range(4)):
            bbox_ok += 1

        # 마스크 면적 vs COCO area(=bbox면적) 비율  → π/4 근처면 area가 bbox면적이란 증거
        mask = polygon_to_mask(poly, h, w, use_convex_hull=False)
        m_area = int(cv2.countNonZero(mask))
        if coco_area > 0:
            area_ratios.append(m_area / coco_area)

        # convex hull 면적 팽창률 (hull이 원폴리곤 대비 얼마나 부풀리는가)
        hull_mask = polygon_to_mask(poly, h, w, use_convex_hull=True)
        hull_area = int(cv2.countNonZero(hull_mask))
        if m_area > 0:
            hull_inflation.append(hull_area / m_area)

    print("📌 [검증1] 설계 로직 (면적최대 + 퇴화필터)")
    print(f"   추출 성공: {extracted}/{n_img}   누락: {len(missing)}  {missing}")
    print(f"   bbox 일치(±2px): {bbox_ok}/{extracted}")
    ar = np.array(area_ratios)
    print(f"   마스크면적/COCO area 비율: 평균 {ar.mean():.4f}  최소 {ar.min():.4f}  최대 {ar.max():.4f}")
    hi = np.array(hull_inflation)
    print(f"   ConvexHull 면적 팽창률: 평균 {hi.mean():.4f}  최대 {hi.max():.4f}  "
          f"(1.0=동일, 클수록 원형태 왜곡)\n")

    # ── [검증 2] 대안 전략 비교 ────────────────────────────
    print("📌 [검증2] 선택 전략 비교 (다중 body 이미지에서 차이 발생)")
    diff_imgs = []
    for im in coco["images"]:
        iid, fn = im["id"], im["file_name"]
        anns = body_by_image.get(iid, [])
        p_max = pick(anns, "max_area", True)
        p_first_nofilter = pick(anns, "first", False)   # 필터X + 첫번째
        a_max = 0 if p_max is None else cv2.contourArea(p_max)
        a_first = 0 if p_first_nofilter is None else cv2.contourArea(p_first_nofilter)
        if abs(a_max - a_first) > 1:
            diff_imgs.append((fn, a_first, a_max))
    if diff_imgs:
        print("   '첫번째+필터없음' vs '면적최대+필터' 결과가 다른 이미지:")
        for fn, a1, a2 in diff_imgs:
            print(f"     {fn[:30]}...  첫번째={a1:.0f}  면적최대={a2:.0f}  "
                  f"→ {'❌ 첫번째는 퇴화/오선택' if a1 < a2 else '동일'}")
    else:
        print("   (차이 없음)")

    # ── [검증 3] 면적필터 유무 비교 ────────────────────────
    print("\n📌 [검증3] 퇴화(area=0/점부족) 폴리곤 필터 효과")
    bad = 0
    for im in coco["images"]:
        anns = body_by_image.get(im["id"], [])
        raw = [a for a in anns if a.get("segmentation")]
        for a in raw:
            poly = a["segmentation"][0]
            if (a.get("area") or 0) <= 0 or len(poly) < 6:
                bad += 1
    print(f"   전체 body 어노테이션 중 퇴화 폴리곤: {bad}개 → 필터로 제거됨")


if __name__ == "__main__":
    main()
