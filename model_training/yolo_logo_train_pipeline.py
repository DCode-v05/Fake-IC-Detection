"""
yolo_logo_train_pipeline.py
--------------------------------------------
End-to-end pipeline to:
1) Create synthetic images from logo templates and backgrounds
2) Generate YOLO-format labels (class x_center y_center w h normalized)
3) Split into train/val folders and write data.yaml
4) Train a YOLOv8 multi-class detector with ultralytics

Usage:
    python yolo_logo_train_pipeline.py

Assumptions:
- You have a folder `Dataset/` containing subfolders for each company:
    Dataset/
        ├── Texas Instruments
        ├── STMicroelectronics
        ├── ON Semiconductor
        ├── NXP Semiconductors
        ├── Microchip
        ├── Infineon
  Each subfolder should contain logo images (PNG/JPG). These are templates.

- Optionally provide a folder `backgrounds/` with chip-top or metallic textures.
  If no backgrounds available, the script auto-generates noisy metallic-like backgrounds.

Outputs:
- dataset_synth/
    ├── images/
    │    ├── train/
    │    ├── val/
    └── labels/
         ├── train/
         ├── val/
- data.yaml (for YOLOv8)
- Triggers YOLOv8 training using ultralytics
"""

import os
import random
import math
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# -------------------------
# CONFIG
# -------------------------
ROOT = Path(__file__).parent.parent  # Project root
DATASET_DIR = ROOT / "Dataset"          # your logo folders (class subfolders)
BACKGROUNDS_DIR = ROOT / "backgrounds"  # optional backgrounds; if missing, script auto-generates
OUT_DIR = ROOT / "dataset_synth"        # output synthetic dataset
IMG_SIZE = (1280, 720)                  # output image size (WxH); adjust for tiny logos use larger W,H
NUM_IMAGES = 600                        # total synthetic images (reduced for faster testing - increase to 1200+ for production)
MAX_LOGOS_PER_IMAGE = 3                 # number of logos to paste per image
MIN_SCALE = 0.03                        # min logo scale relative to image width (for small logos)
MAX_SCALE = 0.18                        # max logo scale relative to image width
TRAIN_SPLIT = 0.85
RANDOM_SEED = 42

# YOLO training params
YOLO_EPOCHS = 60
YOLO_IMG_SIZE = 640
YOLO_MODEL = "yolov8n.pt"  # pretrained base; ultralytics will download if missing

# Create output directories
(images_out := OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
(labels_out := OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)
(images_out / "train").mkdir(exist_ok=True)
(images_out / "val").mkdir(exist_ok=True)
(labels_out / "train").mkdir(exist_ok=True)
(labels_out / "val").mkdir(exist_ok=True)

# -------------------------
# Helper utilities
# -------------------------
def list_logo_classes(dataset_dir: Path):
    """
    Returns list of (class_name, list_of_valid_images)
    Filters out corrupted, SVG, and AVIF files
    """
    classes = []
    for cls in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        valid_imgs = []
        for p in cls.iterdir():
            # Only accept PNG and JPG, skip SVG and AVIF
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                # Verify image can be loaded
                try:
                    test_img = Image.open(p)
                    test_img.verify()
                    test_img.close()
                    valid_imgs.append(p)
                except Exception as e:
                    print(f"⚠️  Skipping corrupted/invalid image: {p.name} - {e}")
        
        if valid_imgs:
            classes.append((cls.name, valid_imgs))
            print(f"✅ Loaded {len(valid_imgs)} valid images for '{cls.name}'")
        else:
            print(f"⚠️  Warning: No valid images found for '{cls.name}'")
    
    return classes

def load_backgrounds(bg_dir: Path):
    bgs = []
    if bg_dir.exists():
        for p in bg_dir.iterdir():
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                bgs.append(Image.open(p).convert("RGB"))
    return bgs

def generate_metallic_background(size=(1280,720)):
    # Create random noise background + gaussian blur + color tint to mimic chip surface
    w,h = size
    # noise
    arr = (np.random.randn(h, w) * 20 + 128).clip(0,255).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB")
    # embossed-like effect
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(0.9)
    # add subtle gradient
    grad = Image.new("L", (1, h))
    for y in range(h):
        grad.putpixel((0,y), int(255 * (0.5 + 0.5 * math.sin(y / (h/6)))))
    grad = grad.resize((w,h))
    img = Image.composite(img, ImageOps.colorize(grad, (80,80,80), (150,150,150)).convert("RGB"), grad)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    return img

def augment_logo(logo_img: Image.Image):
    # Convert to grayscale-ish etched look with random transforms
    # Input: PIL Image (RGB or RGBA)
    img = logo_img.convert("RGBA")
    # resize down/up a bit
    scale_jitter = random.uniform(0.9, 1.1)
    new_w = int(img.width * scale_jitter)
    new_h = int(img.height * scale_jitter)
    img = img.resize((max(1,new_w), max(1,new_h)), Image.LANCZOS)
    # random rotation
    angle = random.uniform(-30, 30)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True)
    # reduce color to single channel (simulate etched monochrome)
    gray = ImageOps.grayscale(img)
    # apply contrast and thresholding sometimes
    if random.random() < 0.6:
        gray = ImageOps.autocontrast(gray, cutoff=random.randint(0,5))
    # occasionally emboss or edge enhance
    if random.random() < 0.35:
        gray = gray.filter(ImageFilter.EDGE_ENHANCE_MORE)
    # add slight blur to simulate mark roughness
    if random.random() < 0.4:
        gray = gray.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5,1.2)))
    # convert back to RGBA where white areas are transparent (so logo blends)
    # We invert so dark strokes remain opaque
    arr = np.array(gray).astype(np.uint8)
    # create alpha where dark strokes are opaque
    alpha = (255 - arr).clip(0,255)
    rgba = np.dstack([arr, arr, arr, alpha])
    return Image.fromarray(rgba)

def paste_logo_on_bg(bg: Image.Image, logo: Image.Image, position, opacity=1.0):
    # logo is RGBA; bg is RGB
    x, y = position
    logo = logo.copy()
    if opacity < 1.0:
        alpha = logo.split()[3].point(lambda p: int(p * opacity))
        logo.putalpha(alpha)
    bg.paste(logo, (x,y), logo)
    return bg

def create_synthetic_image(index, classes, backgrounds):
    """
    Create one synthetic image by pasting 1..MAX_LOGOS_PER_IMAGE logos on a background
    Returns: (filename, list_of_labels) where each label is (class_id, x_center, y_center, w, h) normalized
    """
    # pick background
    if backgrounds and random.random() < 0.9:
        bg = random.choice(backgrounds).copy().resize(IMG_SIZE)
    else:
        bg = generate_metallic_background(IMG_SIZE)

    W, H = IMG_SIZE
    labels = []
    num_logos = random.randint(1, MAX_LOGOS_PER_IMAGE)
    placed_regions = []  # avoid heavy overlaps (basic)
    for _ in range(num_logos):
        # pick a random class and random logo from class
        class_idx = random.randrange(len(classes))
        class_name, logo_list = classes[class_idx]
        logo_path = random.choice(logo_list)
        
        try:
            logo_img = Image.open(logo_path).convert("RGBA")
        except Exception as e:
            print(f"⚠️ Warning: Could not load {logo_path}: {e}")
            continue
            
        logo_aug = augment_logo(logo_img)
        # random scale relative to image width
        scale = random.uniform(MIN_SCALE, MAX_SCALE)
        max_logo_w = int(W * scale)
        # keep aspect ratio
        aspect = logo_aug.width / max(1, logo_aug.height)
        new_w = max(12, min(max_logo_w, logo_aug.width))
        new_h = max(12, int(new_w / aspect))
        logo_resized = logo_aug.resize((new_w, new_h), Image.LANCZOS)
        # pick a position avoiding edges and big overlaps
        attempts = 0
        while attempts < 30:
            x = random.randint(10, W - new_w - 10)
            y = random.randint(10, H - new_h - 10)
            bbox = (x, y, x+new_w, y+new_h)
            # minimal overlap rule
            overlap = False
            for r in placed_regions:
                # IoU approx
                ax1, ay1, ax2, ay2 = bbox
                bx1, by1, bx2, by2 = r
                inter_w = max(0, min(ax2,bx2)-max(ax1,bx1))
                inter_h = max(0, min(ay2,by2)-max(ay1,by1))
                inter_area = inter_w * inter_h
                if inter_area > 0.4 * (new_w * new_h):
                    overlap = True
                    break
            if not overlap:
                break
            attempts += 1
        # paste
        opacity = random.uniform(0.7, 1.0)
        paste_logo_on_bg(bg, logo_resized, (x,y), opacity=opacity)
        placed_regions.append(bbox)
        # compute normalized YOLO format label
        x_center = (x + x + new_w) / 2.0 / W
        y_center = (y + y + new_h) / 2.0 / H
        w_norm = new_w / W
        h_norm = new_h / H
        labels.append((class_idx, x_center, y_center, w_norm, h_norm))

    # minor global noise / vignette to mimic photo conditions
    if random.random() < 0.6:
        enhancer = ImageEnhance.Brightness(bg)
        bg = enhancer.enhance(random.uniform(0.9, 1.05))
    return bg, labels

# -------------------------
# MAIN: build synthetic dataset
# -------------------------
def build_synthetic_dataset():
    random.seed(RANDOM_SEED)
    classes = list_logo_classes(DATASET_DIR)
    assert classes, f"No classes found in {DATASET_DIR}. Check folder structure."

    class_names = [c[0] for c in classes]
    print(f"✓ Detected {len(classes)} classes:", class_names)

    # backgrounds
    backgrounds = load_backgrounds(BACKGROUNDS_DIR)
    print(f"✓ Loaded {len(backgrounds)} background images")

    image_meta = []  # list of tuples (img_path, label_path)
    print(f"\n🎨 Generating {NUM_IMAGES} synthetic images...")
    for i in tqdm(range(NUM_IMAGES), desc="Creating synthetic dataset"):
        img, labels = create_synthetic_image(i, classes, backgrounds)
        img_name = f"syn_{i:05d}.jpg"
        label_name = f"syn_{i:05d}.txt"
        img_path = images_out / img_name
        label_path = labels_out / label_name
        img.save(img_path, quality=90)
        # write yolo labels: class_id x_center y_center w h
        with open(label_path, "w") as f:
            for lbl in labels:
                cls_idx, xc, yc, wn, hn = lbl
                f.write(f"{cls_idx} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
        image_meta.append((img_path, label_path))

    # split train/val
    train_meta, val_meta = train_test_split(image_meta, train_size=TRAIN_SPLIT, random_state=RANDOM_SEED)
    
    print(f"\n📊 Splitting dataset: {len(train_meta)} train, {len(val_meta)} val")
    
    # move files into train/val folders
    for img_path, label_path in tqdm(train_meta, desc="Moving train files"):
        (images_out / "train" / img_path.name).write_bytes(img_path.read_bytes())
        (labels_out / "train" / label_path.name).write_text(label_path.read_text())
    for img_path, label_path in tqdm(val_meta, desc="Moving val files"):
        (images_out / "val" / img_path.name).write_bytes(img_path.read_bytes())
        (labels_out / "val" / label_path.name).write_text(label_path.read_text())

    # Cleanup top-level images/labels (keep only train/val folders)
    for p in images_out.iterdir():
        if p.is_file():
            p.unlink()
    for p in labels_out.iterdir():
        if p.is_file():
            p.unlink()

    # Write names file & data.yaml for YOLO
    names_file = OUT_DIR / "data.names"
    with open(names_file, "w") as f:
        for n in class_names:
            f.write(n + "\n")

    data_yaml = OUT_DIR / "data.yaml"
    data_content = {
        "path": str(OUT_DIR.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names
    }
    # Write YAML in simple format
    import yaml
    with open(data_yaml, "w") as f:
        yaml.dump(data_content, f)

    print(f"\n✓ Synthetic dataset built at: {OUT_DIR}")
    print(f"✓ data.yaml written at: {data_yaml}")
    return data_yaml

# -------------------------
# TRAIN YOLO (ultralytics)
# -------------------------
def train_yolo(data_yaml):
    # dynamic import to delay if ultralytics not installed
    try:
        from ultralytics import YOLO
        import torch
    except Exception as e:
        raise RuntimeError("Install ultralytics (pip install ultralytics) to train YOLOv8.") from e

    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"🚀 Starting YOLOv8 training on {device.upper()}")
    print(f"{'='*60}")
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"⚠️  Running on CPU - training will be slower")
    
    print(f"   Model: {YOLO_MODEL}")
    print(f"   Epochs: {YOLO_EPOCHS}")
    print(f"   Image size: {YOLO_IMG_SIZE}")
    print(f"{'='*60}\n")
    
    # Use a pretrained small yolov8n backbone for speed. Training will fine-tune for classes.
    model = YOLO(YOLO_MODEL)  # this downloads weights if needed
    
    # Training: specify the data yaml, epochs, img size and where to save results
    # GPU-optimized settings
    results = model.train(
        data=str(data_yaml), 
        epochs=YOLO_EPOCHS, 
        imgsz=YOLO_IMG_SIZE, 
        name="yolov8_logo_run",
        patience=10,  # early stopping
        save=True,
        plots=True,
        device=device,  # Explicitly use GPU
        batch=16 if device == 'cuda' else 8,  # Larger batch for GPU
        workers=4 if device == 'cuda' else 2,  # More workers for GPU
        amp=True,  # Automatic Mixed Precision for faster training on GPU
        cache=True,  # Cache images in RAM for faster training
        verbose=True
    )
    
    print(f"\n{'='*60}")
    print(f"✅ YOLO training complete!")
    print(f"{'='*60}")
    print(f"   Check: runs/detect/yolov8_logo_run/")
    print(f"   Best model: runs/detect/yolov8_logo_run/weights/best.pt")
    print(f"   Metrics: runs/detect/yolov8_logo_run/results.png")
    print(f"{'='*60}\n")
    
    return results

# -------------------------
# SIMPLE INFERENCE SNIPPET
# -------------------------
def inference_demo(model_path: str, img_path: str, conf_thres=0.25):
    from ultralytics import YOLO
    import cv2
    
    print(f"\n🔍 Running inference on {img_path}...")
    
    model = YOLO(model_path)
    res = model.predict(source=img_path, conf=conf_thres)[0]
    
    print("\n📊 Detections:")
    for box, conf, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
        x1,y1,x2,y2 = map(int, box.tolist())
        label = model.names[int(cls)]
        print(f"   {label}: conf={float(conf):.3f}, bbox=({x1},{y1},{x2},{y2})")
    
    # draw and show
    im = cv2.imread(img_path)
    for box, conf, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
        x1,y1,x2,y2 = map(int, box.tolist())
        cv2.rectangle(im, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(im, f"{model.names[int(cls)]} {conf:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    
    outp = OUT_DIR / "demo_out.jpg"
    cv2.imwrite(str(outp), im)
    print(f"\n✓ Saved demo output to: {outp}")

# -------------------------
# ENTRY
# -------------------------
if __name__ == "__main__":
    print("="*60)
    print("  YOLOv8 IC Logo Detection - Training Pipeline")
    print("="*60)
    
    print("\n📝 Step 1: Building synthetic dataset...")
    data_yaml = build_synthetic_dataset()
    
    print("\n📝 Step 2: Training YOLOv8 model...")
    train_yolo(data_yaml)
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print("="*60)
    print("\n💡 Next steps:")
    print("   1. Check training metrics in: runs/detect/yolov8_logo_run/")
    print("   2. Use best model: runs/detect/yolov8_logo_run/weights/best.pt")
    print("   3. Test inference with: inference_demo('path/to/best.pt', 'test_image.jpg')")
    print("\n")
