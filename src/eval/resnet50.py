import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split, Dataset
from pathlib import Path

# --- 1. Helper Class for Transforms ---
class ApplyTransform(Dataset):
    """
    Wraps a dataset (or subset) to apply a specific transform on the fly.
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)

# --- 2. Setup Paths & Device ---
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent.parent
data_root = project_root / "data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Data root: {data_root}")

# --- 3. Define Transforms ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- 4. Load & Combine Data (Do this ONCE) ---
print("Loading raw datasets...")
d1 = datasets.Flowers102(root=data_root, split="train", download=False)
d2 = datasets.Flowers102(root=data_root, split="val",   download=False)
d3 = datasets.Flowers102(root=data_root, split="test",  download=False)

full_dataset = ConcatDataset([d1, d2, d3])
total_size = len(full_dataset)
print(f"Total images merged: {total_size}")

# Define Split Sizes (50% / 25% / 25%)
train_size = int(0.50 * total_size)
val_size   = int(0.25 * total_size)
test_size  = total_size - train_size - val_size 

# --- 5. REPEATED SPLIT LOOP ---
# "This random split should be repeated at least twice" 
NUM_RUNS = 2 

for run in range(NUM_RUNS):
    print("\n" + "="*40)
    print(f"STARTING RUN {run + 1}/{NUM_RUNS}")
    print("="*40)

    # A. Perform Random Split with a UNIQUE SEED for each run
    # Changing the seed ensures we get a different random 50/25/25 split every time
    current_seed = 42 + run
    print(f"Splitting data with seed {current_seed}...")
    
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(current_seed)
    )

    # Apply transforms
    train_dataset = ApplyTransform(train_subset, data_transforms['train'])
    val_dataset   = ApplyTransform(val_subset,   data_transforms['val'])
    test_dataset  = ApplyTransform(test_subset,  data_transforms['val'])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=2)

    # B. Fresh Model Initialization
    # We re-initialize the model so weights are reset for the new split
    print("Initializing fresh ResNet50 model...")
    model = models.resnet50(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 102) 
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # C. Training Loop
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f'Run {run+1} - Epoch {epoch+1}/{num_epochs}')
        
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)
        
        print(f'   Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'   Val Loss:   {val_loss:.4f} Acc: {val_acc:.4f}')

    # D. Final Test Evaluation for this Run
    print(f"\n--- Run {run+1} Final Test Evaluation ---")
    model.eval()
    test_loss = 0.0
    test_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)

    final_test_loss = test_loss / len(test_dataset)
    final_test_acc = test_corrects.double() / len(test_dataset)

    print(f'Run {run+1} Test Accuracy: {final_test_acc:.4f}')
    
    # Save model for this specific run
    save_path = project_root / "models" / f"resnet50_run_{run+1}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model for Run {run+1} saved to: {save_path}")

print("\nAll runs complete!")