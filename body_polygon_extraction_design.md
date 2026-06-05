# 수박 Body 폴리곤 추출 로직 설계

`Watermelon.coco` 데이터셋의 COCO 어노테이션에서 **수박 몸통(body) 폴리곤**을 안정적으로 뽑아내는 로직 설계 문서.

---

## 1. 목표

`Watermelon.coco/train/_annotations.coco.json` 에 들어 있는 **정답(ground-truth) 세그멘테이션**으로부터, 각 이미지의 수박 몸통 폴리곤(또는 그 마스크)을 추출한다.

- 기존 `body_extractor.py` 는 YOLO 모델(`best.pt`) **추론**으로 body 마스크를 만든다.
- 본 설계는 그와 달리 **사람이 라벨링한 정답 폴리곤**을 그대로 꺼내 쓰는 로직이다.
- 용도: 학습용 마스크 생성, YOLO 추론 결과 검증(IoU 비교), 특징 추출 파이프라인(`feature_extractor.py`)에 정답 마스크 공급.

---

## 2. 데이터셋 구조 (실측)

```
Watermelon.coco/
├── README.roboflow.txt          # Roboflow COCO export, 121장, 전처리/증강 없음
└── train/
    ├── _annotations.coco.json   # 모든 어노테이션 (약 2.3MB)
    └── *.jpg                     # 수박 사진 121장 (CSV 파일명과 동일)
```

### 2.1 COCO JSON 최상위 키
`info`, `licenses`, `categories`, `images`, `annotations`

### 2.2 카테고리 (`categories`)

| id | name | supercategory | 어노테이션 수 | 비고 |
|----|------|---------------|--------------|------|
| 0 | Watermelon | none | 0 | 부모 클래스(껍데기), 실제 라벨 없음 |
| **1** | **body** | Watermelon | **121** | ⭐ 추출 대상(몸통) |
| 2 | dark green | Watermelon | 1033 | 줄무늬(짙은 녹색) |
| 3 | navel | Watermelon | 2 | 배꼽 (거의 없음) |
| 4 | stem | Watermelon | 82 | 꼭지 |

> **핵심: body 의 `category_id` 는 `1`.** (YOLO `best.pt` 의 class 0 과 번호가 다름에 주의 — 데이터 출처가 다르다.)

### 2.3 이미지 (`images`) 항목 예시
```json
{
  "id": 0,
  "file_name": "7_jpg.rf.r9lFfrqWn6nvSsLYmjZu.jpg",
  "height": 3024, "width": 4032,
  "license": 1,
  "extra": {"name": "7.jpg"}
}
```

### 2.4 어노테이션 (`annotations`) 항목 예시
```json
{
  "id": 1,
  "image_id": 0,
  "category_id": 1,                 // 1 = body
  "bbox": [560, 181, 2565, 2348],   // [x, y, w, h] (좌상단 기준)
  "area": 6022620,
  "iscrowd": 0,
  "segmentation": [[567,1202, 560,1285, 561,1360, ...]]  // 폴리곤 리스트
}
```

- `segmentation` 은 **폴리곤들의 리스트**. 각 폴리곤은 `[x1,y1, x2,y2, ...]` 1차원 배열(평탄화된 좌표).
- 본 데이터셋의 body 는 **전부 단일 폴리곤**(구멍/분리 없음, multi-polygon 0건). 폴리곤 점 개수는 가변(예: 91점).

---

## 3. 엣지 케이스 (실측 — 반드시 처리)

| 케이스 | 발생 이미지 | 내용 | 처리 방침 |
|--------|-------------|------|-----------|
| body **0개** | `100_jpg.rf.OOZp4WadTyy4SSK6Qbro.jpg` | body 어노테이션 누락 | 해당 이미지 **스킵 + 경고 로그** (None 반환) |
| body **2개**(=1개+빈것) | `31_jpg.rf.5DUS8GwKv2FRAM1v9eDT.jpg` | id 1065=정상(261점), id 1066=**빈 어노테이션**(`area=0, bbox=[...,0,8], segmentation=[]`) | **빈 segmentation 제거** 후 1개 선택 |
| 빈/RLE segmentation | 위 31번의 id 1066 | 폴리곤 좌표가 아예 없음(`[]`) | 유효성 필터로 제거 |

> 119장은 body 정확히 1개, 1장은 0개, 1장은 "정상 1개 + 빈 어노테이션 1개". → "이미지당 body 1개"를 가정하면 안 되고, **0개·다수 모두 방어**해야 한다.
> ⚠️ **교차검증으로 밝혀진 사실(§9 참조): 31번의 두 번째는 "작은 폴리곤"이 아니라 `segmentation=[]`(빈 값)이다.** 따라서 진짜 1차 방어선은 *면적 비교*가 아니라 **"빈 segmentation 제거"** 다.

---

## 4. 추출 로직 설계

### 4.1 처리 흐름 (파이프라인)

```
_annotations.coco.json
        │
        ▼
[1] JSON 로드
        │
        ▼
[2] 인덱스 맵 구축
    - image_id → 이미지 메타(file_name, w, h)
    - image_id → [body 어노테이션...]   (category_id == 1 만)
        │
        ▼
[3] 이미지별 body 어노테이션 수집
        │
        ▼
[4] 유효성 필터       (점 ≥ 3, area > 0)
        │
        ▼
[5] body 선택         (여러 개면 area 최대 1개)
        │
        ├── 없음 → None + 경고  (예: 100번)
        ▼
[6] segmentation → 폴리곤 좌표 변환
    [x1,y1,x2,y2,...] → (N,2) ndarray
        │
        ▼
[7] (선택) 산출물 생성
    - 폴리곤 좌표 그대로
    - 또는 마스크 래스터화 (fillPoly)
    - 또는 Convex Hull 마스크 (기존 파이프라인 호환)
```

### 4.2 단계별 상세

**[1] JSON 로드**
- `json.load(open(path, encoding='utf-8'))`.

**[2] 인덱스 맵 구축** (O(N) 1회 순회)
- `images_by_id = {im['id']: im for im in coco['images']}`
- `body_by_image = defaultdict(list)`; `annotations` 순회하며 `category_id == BODY_ID(1)` 인 것만 `body_by_image[a['image_id']]` 에 append.
- `category_id` 는 하드코딩(1) 대신 `categories` 에서 `name == 'body'` 로 동적 조회 권장(데이터셋 바뀌어도 안전).

**[4] 유효성 필터**
- 각 폴리곤: 좌표 길이가 짝수이고 `len//2 >= 3` 인지, `area > 0` 인지 확인.
- `segmentation` 이 비었거나 RLE(dict) 형식이면 제외(본 데이터셋은 폴리곤만 존재).

**[5] body 선택**
- 후보가 여러 개면 **실제 폴리곤 면적(`cv2.contourArea`)** 가장 큰 것 1개.
  - ⚠️ COCO `area` 필드를 면적 기준으로 쓰지 말 것 — **교차검증 결과 이 필드는 폴리곤 면적이 아니라 bbox 면적(`w×h`)** 이다(§9). 가늘고 긴 오브젝트면 bbox 면적이 실제보다 과대평가될 수 있으므로 `contourArea`가 안전하다.
- 후보 0개면 `(None, "body 어노테이션 없음")` 반환.

**[6] 폴리곤 좌표 변환**
- `flat = seg[0]` → `np.array(flat).reshape(-1, 2)` → `(N,2)` 정수 좌표.

**[7] 산출물(용도별 선택)**
- **(a) 폴리곤 좌표**: 그대로 반환 → 면적/형태 분석, 시각화.
- **(b) 채운 마스크**: `np.zeros((h,w),uint8)` 에 `cv2.fillPoly([poly], 255)`.
- **(c) Convex Hull 마스크**: `cv2.convexHull(poly)` 후 fill → **기존 `body_extractor.py` 와 동일 형식**이라 `feature_extractor.calculate_watermelon_features()` 에 바로 투입 가능.

### 4.3 인터페이스(시그니처) 제안

```python
# coco_body_extractor.py (설계안)

BODY_CATEGORY_NAME = "body"   # category_id 는 이 이름으로 조회

def load_coco(json_path: str) -> dict: ...

def build_indexes(coco: dict):
    """returns (images_by_id, body_anns_by_image, body_category_id)"""

def get_body_polygon(coco_indexes, image_id) -> np.ndarray | None:
    """이미지의 body 폴리곤 (N,2). 없으면 None.
       여러 개면 area 최대, 퇴화 폴리곤 제외."""

def polygon_to_mask(polygon, height, width, use_convex_hull=False) -> np.ndarray:
    """폴리곤 → uint8 마스크(0/255). 기존 파이프라인 호환은 use_convex_hull=True."""

def iter_body_masks(json_path, image_dir, use_convex_hull=False):
    """모든 이미지에 대해 (file_name, mask 또는 None) 제너레이터."""
```

### 4.4 핵심 의사코드

```python
def get_body_polygon(body_anns_by_image, image_id):
    candidates = []
    for a in body_anns_by_image.get(image_id, []):
        seg = a.get("segmentation")
        if not seg or isinstance(seg, dict):      # 빈 segmentation / RLE 제외 ← 31번 1차 방어선
            continue
        poly = seg[0]
        if len(poly) < 6:                         # 점 3개 미만(=좌표 6개) 제외
            continue
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        true_area = cv2.contourArea(pts)          # COCO 'area' 대신 실제 폴리곤 면적
        if true_area <= 0:                         # 퇴화 폴리곤 제외(안전망)
            continue
        candidates.append((true_area, pts))

    if not candidates:
        return None                               # 100번 같은 케이스
    _, best_poly = max(candidates, key=lambda t: t[0])   # 실제 면적 최대 선택
    return best_poly
```

---

## 5. 산출물 형식 (출력 스펙)

| 형식 | 타입 | 용도 |
|------|------|------|
| 폴리곤 좌표 | `np.ndarray (N,2) int32` | 형태 분석, 다른 포맷(YOLO txt 등) 변환 |
| 채운 마스크 | `np.ndarray (H,W) uint8 {0,255}` | 영역 추출, 픽셀 통계 |
| Convex Hull 마스크 | `np.ndarray (H,W) uint8 {0,255}` | **`feature_extractor` 직접 입력** |

---

## 6. 기존 파이프라인과의 연계

- `feature_extractor.calculate_watermelon_features(img_bgr, solid_body_mask)` 는 **`solid_body_mask`(uint8 0/255)** 를 입력으로 받는다.
- 따라서 `polygon_to_mask(..., use_convex_hull=True)` 산출물을 그대로 넘기면, **YOLO 추론 대신 정답 마스크**로 특징 A·B·C 를 다시 계산할 수 있다.
- 활용 시나리오:
  1. **정답 기반 특징 재추출** → 라벨 품질 좋은 특징으로 회귀 모델 재학습.
  2. **YOLO 검증** → `best.pt` 추론 마스크 vs 정답 마스크 **IoU** 측정으로 세그멘테이션 성능 평가.

---

## 7. 검증(테스트) 항목  — `validate_body_extraction.py` 로 자동 검증

- [x] 121장 중 **120장**에서 body 폴리곤 정상 추출, **100번만 None** → ✅ 통과
- [x] `31번`이 빈 어노테이션이 아니라 **정상 폴리곤(261점)** 을 선택 → ✅ 통과
- [x] 추출 폴리곤의 bbox 가 어노테이션 `bbox` 와 일치(±2px) → ✅ **120/120 일치**
- [x] ~~마스크 픽셀 면적이 `area` 값과 근사~~ → ❌ **`area`는 bbox 면적이라 마스크면적/area ≈ 0.789(=π/4)**. 검증 기준을 "마스크면적 ≈ `area` × π/4" 또는 "`contourArea(poly)`와 일치"로 교체.
- [ ] 마스크가 이미지 경계(H, W) 를 벗어나지 않는지.

---

## 8. 주의사항

- **category_id 매핑**: body=1 은 이 export 기준값. 재export 시 번호가 바뀔 수 있으므로 **이름(`body`)으로 조회**할 것.
- **좌표계**: COCO 폴리곤은 픽셀 절대좌표(원본 해상도, 예: 4032×3024). 리사이즈된 이미지에 쓰려면 동일 배율로 스케일 필요.
- **`bbox` 포맷**: COCO 는 `[x, y, w, h]`(좌상단+크기). OpenCV `boundingRect` 와 같은 포맷이나, `[x1,y1,x2,y2]` 로 착각 금지.
- **iscrowd**: 본 데이터셋 body 는 전부 `iscrowd=0`(폴리곤). RLE 분기는 방어적으로만 둔다.

---

## 9. 교차검증 결과 (`validate_body_extraction.py` 실측)

설계 로직이 **"가장 body를 잘 뽑는 로직"인지** 121장 전체로 검증하고, 대안 전략과 비교했다.

### 9.1 설계 로직 성능

| 항목 | 결과 | 판정 |
|------|------|------|
| 추출 성공 | **120/121** (100번만 누락=정답) | ✅ 완벽 |
| bbox 일치(±2px) | **120/120** | ✅ 좌표 변환 정확 |
| 마스크면적 / COCO `area` | 평균 **0.789** (0.772~0.829) | ⚠️ `area`=bbox면적이라는 증거 (π/4≈0.785) |
| ConvexHull 면적 팽창률 | 평균 **1.0016** (최대 1.0043) | ✅ 무시 가능 |

### 9.2 핵심 발견 (설계 보정 사유)

1. **COCO `area` 필드 = bbox 면적(`w×h`)** — 전 어노테이션에서 `area / (w×h) = 1.0000`. 폴리곤 실면적이 아니다.
   → 면적 기준 선택·검증은 **`cv2.contourArea(polygon)`** 로 해야 정확. (§4.2 [5], §7 반영)

2. **31번의 "두 번째 body"는 빈 어노테이션** — `segmentation=[]`, `bbox=[870,3424,0,8]`, `area=0`. 작은 폴리곤이 아니라 **좌표 자체가 없음**.
   → 1차 방어선은 "면적 비교"가 아니라 **"빈 segmentation 제거"**. (§3 반영)

3. **ConvexHull vs 원폴리곤 차이 0.16%** — 수박 body 라벨이 이미 볼록에 가까워, hull을 써도 형태 왜곡이 거의 없다.
   → 기존 `body_extractor.py` 의 convex hull 방식을 **그대로 써도 안전**. 정밀도가 중요하면 원폴리곤, 파이프라인 호환이 중요하면 hull.

### 9.3 대안 전략 비교

| 전략 | 121장 결과 | 평가 |
|------|-----------|------|
| **면적최대 + 빈seg/퇴화 필터** (설계) | 120장 정확 추출 | ✅ 채택 |
| 첫번째 + 필터없음 | 본 데이터셋에선 동일 결과 | ⚠️ 우연히 일치 (31번 빈 어노테이션이 `segmentation` 검사에서 걸러지기 때문). 빈 값이 아니라 *작은 유효 폴리곤*이 섞이면 깨짐 → 비권장 |

> **결론: 설계 로직(빈 segmentation 제거 → 실면적 최대 선택)은 현 데이터셋에서 최적이며, `area` 필드 대신 `contourArea` 사용·"빈 seg 제거 우선" 두 가지 보정으로 미래 데이터에도 견고하다.**
```
