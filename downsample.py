import shutil
import random
from pathlib import Path

# Paths
src_dir = Path("data/mnist256/all")        # original 70k images
dst_dir = Path("data/mnist256/downsampled")  # new folder for 7k images
dst_dir.mkdir(parents=True, exist_ok=True)

# List all images in the source directory
all_images = list(src_dir.glob("*.png"))

# Sample 7000 images
sample_size = 7000
sampled_images = random.sample(all_images, sample_size)

# Copy sampled images to the destination folder
for img_path in sampled_images:
    shutil.copy(img_path, dst_dir / img_path.name)

print(f"Copied {len(sampled_images)} images to {dst_dir}")