from torchvision.datasets import Flowers102
from torchvision import transforms

# We usually need some basic transform to convert images to tensors
transform = transforms.Compose([transforms.ToTensor()])

# Load all three splits (this won't re-download, just load existing data)
train_set = Flowers102(root="data", split="train", transform=transform, download=True)
val_set   = Flowers102(root="data", split="val",   transform=transform, download=True)
test_set  = Flowers102(root="data", split="test",  transform=transform, download=True)

print("--- Data Check ---")
print(f"Training images:   {len(train_set)} (Should be 1,020)")
print(f"Validation images: {len(val_set)}   (Should be 1,020)")
print(f"Test images:       {len(test_set)}  (Should be 6,149)")