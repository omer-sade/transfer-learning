import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split, Dataset
from pathlib import Path
import copy 

from utils import EarlyStopper, ApplyTransform


# --- Setup Paths & Device ---
current_script_path = Path(__file__).resolve()
# Assuming the script is 3 levels deep from root based on your previous code
project_root = current_script_path.parent.parent.parent 
data_root = project_root / "data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Data root: {data_root}")

# --- Define Transforms ---
# ViT usually expects 224x224.
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # ImageNet mean/std normalization
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- Load & Combine Data ---
print("Loading raw datasets...")
d1 = datasets.Flowers102(root=data_root, split="train", download=True)
d2 = datasets.Flowers102(root=data_root, split="val",   download=True)
d3 = datasets.Flowers102(root=data_root, split="test",  download=True)

full_dataset = ConcatDataset([d1, d2, d3])
total_size = len(full_dataset)
print(f"Total images merged: {total_size}")

# Define Split Sizes (50% / 25% / 25%)
train_size = int(0.50 * total_size)
val_size   = int(0.25 * total_size)
test_size  = total_size - train_size - val_size 

# --- REPEATED SPLIT LOOP ---
NUM_RUNS = 2 

BATCH_SIZE = 16

for run in range(NUM_RUNS):
    print("\n" + "="*40)
    print(f"STARTING RUN {run + 1}/{NUM_RUNS}")
    print("="*40)

    # A. Perform Random Split with a UNIQUE SEED
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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # B. Fresh Model Initialization
    print("Initializing fresh ViT_b_16 model...")
    # 'weights="DEFAULT"' loads the best available ImageNet weights
    model = models.vit_b_16(weights='DEFAULT')
    
    # MODIFYING THE HEAD FOR VIT
    num_ftrs = model.heads.head.in_features 
    model.heads = nn.Linear(num_ftrs, 102) 
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # Initialize Early Stopper and Best Model Tracking
    early_stopper = EarlyStopper(patience=3, min_delta=0.00)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # C. Training Loop
    # Increased epochs because early stopping will handle the limit
    num_epochs = 20 
    
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

        # --- TRACK BEST MODEL ---
        # Track Best Acc
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            # Optional: Print that we found a new best
            # print(f"   * New Best Model found (Acc: {val_acc:.4f}) *")

        # --- EARLY STOPPING ---
        if early_stopper.early_stop(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # D. Final Test Evaluation using BEST model
    print(f"\n--- Run {run+1} Final Test Evaluation (Loading Best Model) ---")
    
    # LOAD THE BEST WEIGHTS BEFORE TESTING
    model.load_state_dict(best_model_wts)
    
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
    
    # Save model (saving the best one)
    save_path = project_root / "models" / f"vit_b_run_{run+1}.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), save_path)
    print(f"Best Model for Run {run+1} saved to: {save_path}")

print("\nAll runs complete!")