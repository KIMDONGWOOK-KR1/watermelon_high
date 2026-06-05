import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# 💡 테스트할 수박 사진 경로
IMAGE_PATH = 'test3.jpg'  # 여기에 원하는 이미지 경로를 입력하세요!

# 1. 이미지 로드
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    print("❌ 이미지를 찾을 수 없습니다.")
    exit()

# Matplotlib은 RGB를 쓰므로 변환
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2. 파이프라인과 동일한 조명 보정(CLAHE) 적용
lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(l)
img_clahe_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
hsv_img = cv2.cvtColor(img_clahe_bgr, cv2.COLOR_BGR2HSV)

# 3. 화면(Figure) 세팅
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.45) # 슬라이더가 들어갈 하단 여백 확보
fig.canvas.manager.set_window_title('HSV Tuner')

# 초기 마스킹 결과
lower_bound = np.array([25, 30, 0])
upper_bound = np.array([85, 255, 125])
mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

# 이미지 띄우기
img_display = ax.imshow(result)
ax.set_title("🍉 HSV 줄무늬 정밀 튜너", fontsize=16)
ax.axis('off')

# 4. 슬라이더 UI 축 생성
axcolor = 'lightgoldenrodyellow'
ax_hmin = plt.axes([0.15, 0.35, 0.65, 0.03], facecolor=axcolor)
ax_hmax = plt.axes([0.15, 0.30, 0.65, 0.03], facecolor=axcolor)
ax_smin = plt.axes([0.15, 0.25, 0.65, 0.03], facecolor=axcolor)
ax_smax = plt.axes([0.15, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_vmin = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_vmax = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)

# 슬라이더 객체 생성
s_hmin = Slider(ax_hmin, 'H Min (색상)', 0, 179, valinit=25, valstep=1)
s_hmax = Slider(ax_hmax, 'H Max (색상)', 0, 179, valinit=85, valstep=1)
s_smin = Slider(ax_smin, 'S Min (채도)', 0, 255, valinit=30, valstep=1)
s_smax = Slider(ax_smax, 'S Max (채도)', 0, 255, valinit=255, valstep=1)
s_vmin = Slider(ax_vmin, 'V Min (밝기)', 0, 255, valinit=0, valstep=1)
s_vmax = Slider(ax_vmax, 'V Max (밝기)', 0, 255, valinit=125, valstep=1)

# 5. 슬라이더를 움직일 때마다 실행되는 업데이트 함수
def update(val):
    lower = np.array([s_hmin.val, s_smin.val, s_vmin.val])
    upper = np.array([s_hmax.val, s_smax.val, s_vmax.val])
    
    current_mask = cv2.inRange(hsv_img, lower, upper)
    current_result = cv2.bitwise_and(img_rgb, img_rgb, mask=current_mask)
    
    img_display.set_data(current_result)
    fig.canvas.draw_idle()

# 슬라이더에 업데이트 함수 연결
s_hmin.on_changed(update)
s_hmax.on_changed(update)
s_smin.on_changed(update)
s_smax.on_changed(update)
s_vmin.on_changed(update)
s_vmax.on_changed(update)

# 6. 코드 복사 출력 버튼
ax_btn = plt.axes([0.8, 0.02, 0.15, 0.05])
btn = Button(ax_btn, 'Print Code', color='lightblue', hovercolor='0.975')

def print_code(event):
    print("\n✅ [최종 선택된 HSV 값 - 복사해서 코드에 붙여넣으세요!]")
    print(f"lower_green = np.array([{int(s_hmin.val)}, {int(s_smin.val)}, {int(s_vmin.val)}])")
    print(f"upper_green = np.array([{int(s_hmax.val)}, {int(s_smax.val)}, {int(s_vmax.val)}])")

btn.on_clicked(print_code)

print("🎛️ HSV 튜너 창이 열립니다. 슬라이더를 움직여 최적의 범위를 찾으세요!")
plt.show()