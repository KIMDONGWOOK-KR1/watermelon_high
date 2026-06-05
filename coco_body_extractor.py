# coco_body_extractor.py
# Watermelon.coco 의 정답(GT) 어노테이션에서 수박 body 폴리곤/마스크를 추출한다.
# 로직 근거: body_polygon_extraction_design.md (validate_body_extraction.py 로 교차검증된 버전)
#   - 카테고리는 '이름'(body)으로 조회
#   - 빈 segmentation / 점<3 / 면적<=0 폴리곤 제거
#   - 여러 개면 '실제 폴리곤 면적(contourArea)' 최대 1개 선택

import json
import os
from collections import defaultdict

import cv2
import numpy as np

BODY_NAME = "body"


def load_coco(json_path):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def build_indexes(coco):
    """returns (images_by_id, images_by_name, body_by_image, body_category_id)"""
    images_by_id = {im["id"]: im for im in coco["images"]}
    images_by_name = {im["file_name"]: im for im in coco["images"]}
    body_id = next(c["id"] for c in coco["categories"] if c["name"] == BODY_NAME)
    body_by_image = defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] == body_id:
            body_by_image[a["image_id"]].append(a)
    return images_by_id, images_by_name, body_by_image, body_id


def get_body_polygon(body_by_image, image_id):
    """이미지의 body 폴리곤 (N,2) int32. 없으면 None.
       빈 seg/퇴화 폴리곤 제거 후, 실제 면적 최대 폴리곤 선택."""
    candidates = []
    for a in body_by_image.get(image_id, []):
        seg = a.get("segmentation")
        if not seg or isinstance(seg, dict):       # 빈 segmentation / RLE 제외
            continue
        flat = seg[0]
        if len(flat) < 6:                          # 점 3개 미만 제외
            continue
        pts = np.array(flat, dtype=np.int32).reshape(-1, 2)
        area = cv2.contourArea(pts)                # COCO 'area'(=bbox면적) 대신 실면적
        if area <= 0:                              # 퇴화 폴리곤 제외
            continue
        candidates.append((area, pts))

    if not candidates:
        return None
    _, best = max(candidates, key=lambda t: t[0])
    return best


def polygon_to_mask(polygon, height, width, use_convex_hull=False):
    """폴리곤 (N,2) → uint8 마스크(0/255).
       use_convex_hull=True 면 기존 body_extractor.py 와 동일한 볼록껍질 마스크."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if polygon is None:
        return mask
    pts = cv2.convexHull(polygon) if use_convex_hull else polygon
    cv2.fillPoly(mask, [pts], 255)
    return mask


def get_body_mask_by_filename(coco_indexes, file_name, use_convex_hull=False):
    """파일명으로 body 마스크 반환. (mask, error). body 없으면 (None, 사유)."""
    images_by_id, images_by_name, body_by_image, _ = coco_indexes
    im = images_by_name.get(file_name)
    if im is None:
        return None, "이미지가 COCO에 없음"
    poly = get_body_polygon(body_by_image, im["id"])
    if poly is None:
        return None, "body 어노테이션 없음"
    mask = polygon_to_mask(poly, im["height"], im["width"], use_convex_hull)
    return mask, None


def iter_body_masks(json_path, use_convex_hull=False):
    """(file_name, mask 또는 None) 제너레이터 — 전체 이미지 순회."""
    coco = load_coco(json_path)
    idx = build_indexes(coco)
    images_by_id, _, body_by_image, _ = idx
    for im in coco["images"]:
        poly = get_body_polygon(body_by_image, im["id"])
        mask = polygon_to_mask(poly, im["height"], im["width"], use_convex_hull) if poly is not None else None
        yield im["file_name"], mask


if __name__ == "__main__":
    # 간단 자체 점검
    JSON = "Watermelon.coco/train/_annotations.coco.json"
    coco = load_coco(JSON)
    idx = build_indexes(coco)
    ok = sum(1 for _, m in iter_body_masks(JSON) if m is not None)
    total = len(coco["images"])
    print(f"body 마스크 추출: {ok}/{total} (None {total - ok}장)")
