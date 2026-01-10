import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Extracted from your Logs ---
epochs = [1, 2, 3, 4, 5]

# Run 1 Data
r1_train_loss = [3.1274, 0.6084, 0.1315, 0.0486, 0.0260]
r1_val_loss   = [1.2062, 0.2931, 0.1796, 0.1439, 0.1223]
r1_train_acc  = [0.3940, 0.8964, 0.9836, 0.9954, 0.9973]
r1_val_acc    = [0.7596, 0.9438, 0.9570, 0.9604, 0.9653]

# Run 2 Data
r2_train_loss = [3.1790, 0.6170, 0.1318, 0.0521, 0.0268]
r2_val_loss   = [1.1766, 0.2975, 0.1887, 0.1397, 0.1306]
r2_train_acc  = [0.3813, 0.9018, 0.9836, 0.9963, 0.9976]
r2_val_acc    = [0.7728, 0.9355, 0.9590, 0.9638, 0.9638]

# --- 2. Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Graph 1: Cross-Entropy Loss
# -----------------------------
ax1.plot(epochs, r1_train_loss, 'b-o', label='Run 1 Train')
ax1.plot(epochs, r1_val_loss,   'b--o', label='Run 1 Val')
ax1.plot(epochs, r2_train_loss, 'r-s', label='Run 2 Train')
ax1.plot(epochs, r2_val_loss,   'r--s', label='Run 2 Val')

ax1.set_title('Cross-Entropy Loss per Epoch - ResNet50')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_xticks(epochs)
ax1.grid(True)
ax1.legend()

# Graph 2: Accuracy
# -----------------------------
ax2.plot(epochs, r1_train_acc, 'b-o', label='Run 1 Train')
ax2.plot(epochs, r1_val_acc,   'b--o', label='Run 1 Val')
# Plot Run 1 Test as a horizontal line

ax2.plot(epochs, r2_train_acc, 'r-s', label='Run 2 Train')
ax2.plot(epochs, r2_val_acc,   'r--s', label='Run 2 Val')
# Plot Run 2 Test as a horizontal line

ax2.set_title('Accuracy per Epoch - ResNet50')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.set_xticks(epochs)
ax2.grid(True)
ax2.legend()

# --- 3. Save ---
plt.tight_layout()
filename = "training_results.png"
plt.savefig(filename)
print(f"Graph saved to {filename}")