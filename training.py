import os
import shutil
from PIL import Image
import random
import glob

# === Paths ===
base_path = "/mnt/sda1/FYP_2024/Helitha/CCTV/UA-DETRAC"
train_image_path = os.path.join(base_path, "images/train")
val_image_path   = os.path.join(base_path, "images/val")
test_image_path  = os.path.join(base_path, "images/test")
train_label_path = os.path.join(base_path, "labels/train")
val_label_path   = os.path.join(base_path, "labels/val")
test_label_path  = os.path.join(base_path, "labels/test")
calib_image_path = os.path.join(base_path, "images/calib")
calib_label_path = os.path.join(base_path, "labels/calib")

# === Ensure target directories exist ===
for path in [test_image_path, test_label_path, calib_image_path, calib_label_path]:
    os.makedirs(path, exist_ok=True)

# === Train / Val Image Stats ===
train_images = [f for f in os.listdir(train_image_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
val_images = sorted([f for f in os.listdir(val_image_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

print(f"Number of training images: {len(train_images)}")
print(f"Number of validation images (before split): {len(val_images)}")

# === Sample Resolution ===
sample_image = train_images[0]
sample_path = os.path.join(train_image_path, sample_image)
with Image.open(sample_path) as img:
    width, height = img.size
    print(f"Resolution of sample image ({sample_image}): {width}x{height}")

# === Move 50% of val to test ===
test_split = int(len(val_images) * 0.5)
test_files = val_images[:test_split]

for fname in test_files:
    src_img = os.path.join(val_image_path, fname)
    dst_img = os.path.join(test_image_path, fname)
    shutil.move(src_img, dst_img)

    label_name = os.path.splitext(fname)[0] + ".txt"
    src_lbl = os.path.join(val_label_path, label_name)
    dst_lbl = os.path.join(test_label_path, label_name)
    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)

print(f" Moved {len(test_files)} val images and labels to test set.")

# === Create 500-image Calibration Set ===
remaining_val_images = sorted(
    glob.glob(os.path.join(val_image_path, "*.jpg")) +
    glob.glob(os.path.join(val_image_path, "*.png"))
)

num_calib = min(500, len(remaining_val_images))
calib_images = random.sample(remaining_val_images, num_calib)

for img_path in calib_images:
    img_name = os.path.basename(img_path)
    lbl_name = os.path.splitext(img_name)[0] + ".txt"

    shutil.copy(img_path, os.path.join(calib_image_path, img_name))

    src_lbl = os.path.join(val_label_path, lbl_name)
    if os.path.isfile(src_lbl):
        shutil.copy(src_lbl, os.path.join(calib_label_path, lbl_name))

print(f" Copied {len(calib_images)} images (+ labels if present) to calibration set.")

# === Final image counts ===
def count_images(folder):
    return len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

print("\n Final counts:")
print(f"Train images: {count_images(train_image_path)}")
print(f"Val images:   {count_images(val_image_path)}")
print(f"Test images:  {count_images(test_image_path)}")
print(f"Calib images: {count_images(calib_image_path)}")


# === Fine Tune with Ultralytics YOLO ===
from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolo11m.pt")

# Fine-tune the model
results = model.train(
    data="/mnt/sda1/FYP_2024/Helitha/CCTV/ua_detrac.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    workers=4,
    patience=10,
    freeze=10,
    # Augmentation options
    degrees=10,      # rotation
    translate=0.1,   # translation
    scale=0.5,       # scale range
    shear=2,         # shear
    perspective=0.001,
    flipud=0.5,      # vertical flip prob
    fliplr=0.5,      # horizontal flip prob
    mosaic=1.0,      # mosaic aug
    mixup=0.1,       # mixup aug
    hsv_h=0.015,     # HSV hue aug
    hsv_s=0.7,       # HSV saturation aug
    hsv_v=0.4,        # HSV value aug (brightness)
    device='cuda:0',  # Use GPU if available
)
