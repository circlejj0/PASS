# 노트북 내장카메라로 실시간 화면을 YOLO를 통해 추론
from ultralytics import YOLO
import cv2

# 모델 로드
model = YOLO("yolo11n-seg.pt")

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 내장카메라: 0, 외장카메라: 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 추론
    results = model.predict(source=frame, conf=0.5, save=False, show=False)

    # 결과를 OpenCV용으로 시각화
    annotated_frame = results[0].plot()

    # 창으로 표시
    cv2.imshow("YOLO11n-seg Webcam", annotated_frame)

    # q 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
