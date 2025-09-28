import os
import random
import shutil

# 원본 데이터셋
src_dir = "/home/circle_jj/ros2_ws/src/yolo/train/images"

# 새로운 폴더
train_dir = "/home/circle_jj/ros2_ws/src/yolo/train_split/images"
val_dir   = "/home/circle_jj/ros2_ws/src/yolo/valid_split/images"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 이미지 파일 리스트
files = [f for f in os.listdir(src_dir) if f.endswith((".jpg", ".png"))]
random.shuffle(files)

# 비율 설정 (예: 80% train, 20% val)
split_ratio = 0.8
split_idx = int(len(files) * split_ratio)

train_files = files[:split_idx]
val_files = files[split_idx:]

# 파일 이동
for f in train_files:
    shutil.copy(os.path.join(src_dir, f), os.path.join(train_dir, f))
for f in val_files:
    shutil.copy(os.path.join(src_dir, f), os.path.join(val_dir, f))

print(f"Train: {len(train_files)} images, Val: {len(val_files)} images")
