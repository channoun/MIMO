import torch
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Output directory for all images
output_root = Path("data/mnist256/all")
output_root.mkdir(parents=True, exist_ok=True)

# Transform: resize to 256x256 and convert to RGB
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3)  # convert to RGB
])

# Download MNIST train + test
mnist_train = datasets.MNIST(root="data/mnist", train=True, download=True)
mnist_test  = datasets.MNIST(root="data/mnist", train=False, download=True)

# Combine train + test
data = torch.cat([mnist_train.data, mnist_test.data], dim=0)
targets = torch.cat([mnist_train.targets, mnist_test.targets], dim=0)

# Save images
for i, img_array in enumerate(tqdm(data, desc="Saving MNIST images")):
    img = Image.fromarray(img_array.numpy(), mode="L")
    img = transform(img)
    img_path = output_root / f"{i}_{targets[i].item()}.png"
    img.save(img_path)

print(f"MNIST dataset prepared at {output_root}, total images: {len(data)}")