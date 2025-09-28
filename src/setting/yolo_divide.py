import os
import random
import shutil

# 원본 데이터셋
images_dir = "/home/circle_jj/ros2_ws/src/yolo/train/images"
labels_dir = "/home/circle_jj/ros2_ws/src/yolo/train/labels"

# 새로 만들 폴더
train_images = "/home/circle_jj/ros2_ws/src/yolo/train_split/images"
train_labels = "/home/circle_jj/ros2_ws/src/yolo/train_split/labels"
val_images   = "/home/circle_jj/ros2_ws/src/yolo/valid_split/images"
val_labels   = "/home/circle_jj/ros2_ws/src/yolo/valid_split/labels"

for d in [train_images, train_labels, val_images, val_labels]:
    os.makedirs(d, exist_ok=True)

# 이미지 파일 리스트
files = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))]
random.shuffle(files)

split_ratio = 0.8
split_idx = int(len(files) * split_ratio)

train_files = files[:split_idx]
val_files   = files[split_idx:]

# 이미지와 라벨 같이 복사
for f in train_files:
    shutil.copy(os.path.join(images_dir, f), os.path.join(train_images, f))
    shutil.copy(os.path.join(labels_dir, f.replace(".jpg", ".txt")), os.path.join(train_labels, f.replace(".jpg", ".txt")))

for f in val_files:
    shutil.copy(os.path.join(images_dir, f), os.path.join(val_images, f))
    shutil.copy(os.path.join(labels_dir, f.replace(".jpg", ".txt")), os.path.join(val_labels, f.replace(".jpg", ".txt")))

print(f"Train: {len(train_files)} images, Val: {len(val_files)} images")
