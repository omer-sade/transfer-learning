import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Extracted from your Logs (ViT-B-16) ---
epochs = list(range(1, 21))

# Run 1 Data
r1_train_loss = [3.3727, 1.4653, 0.7015, 0.3602, 0.1986, 0.1203, 0.0795, 0.0576, 0.0427, 0.0345, 0.0275, 0.0223, 0.0193, 0.0157, 0.0135, 0.0118, 0.0102, 0.0090, 0.0079, 0.0070]
r1_val_loss   = [2.1908, 1.1686, 0.7046, 0.4989, 0.3838, 0.3112, 0.2669, 0.2346, 0.2242, 0.2005, 0.1868, 0.1763, 0.1652, 0.1607, 0.1527, 0.1525, 0.1423, 0.1430, 0.1365, 0.1344]
r1_train_acc  = [0.3671, 0.8603, 0.9707, 0.9924, 0.9978, 0.9990, 0.9993, 0.9998, 0.9998, 0.9998, 0.9998, 0.9995, 0.9995, 0.9998, 0.9995, 0.9995, 0.9998, 0.9998, 0.9998, 0.9998]
r1_val_acc    = [0.6771, 0.8964, 0.9492, 0.9634, 0.9668, 0.9702, 0.9741, 0.9756, 0.9726, 0.9775, 0.9785, 0.9766, 0.9775, 0.9780, 0.9780, 0.9775, 0.9785, 0.9780, 0.9780, 0.9780]


# Run 2 Data
r2_train_loss = [3.3577, 1.4516, 0.6953, 0.3609, 0.2064, 0.1225, 0.0811, 0.0568, 0.0427, 0.0330, 0.0261, 0.0213, 0.0175, 0.0147, 0.0125, 0.0106, 0.0091, 0.0079, 0.0068, 0.0060]
r2_val_loss   = [2.1629, 1.1620, 0.7023, 0.5021, 0.3910, 0.3311, 0.2764, 0.2473, 0.2211, 0.2066, 0.1943, 0.1813, 0.1751, 0.1680, 0.1594, 0.1544, 0.1508, 0.1466, 0.1424, 0.1381]
r2_train_acc  = [0.3683, 0.8627, 0.9702, 0.9927, 0.9978, 0.9990, 0.9998, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
r2_val_acc    = [0.6932, 0.8959, 0.9458, 0.9648, 0.9687, 0.9687, 0.9717, 0.9731, 0.9751, 0.9722, 0.9736, 0.9751, 0.9751, 0.9761, 0.9766, 0.9766, 0.9741, 0.9756, 0.9761, 0.9770]

# --- 2. Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Graph 1: Cross-Entropy Loss
# -----------------------------
ax1.plot(epochs, r1_train_loss, 'b-o', label='Run 1 Train')
ax1.plot(epochs, r1_val_loss,   'b--o', label='Run 1 Val')
ax1.plot(epochs, r2_train_loss, 'r-s', label='Run 2 Train')
ax1.plot(epochs, r2_val_loss,   'r--s', label='Run 2 Val')

ax1.set_title('Cross-Entropy Loss per Epoch - ViT-B-16')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_xticks(range(0, 21, 2)) # Tick every 2 epochs for clarity
ax1.grid(True)
ax1.legend()

# Graph 2: Accuracy
# -----------------------------
ax2.plot(epochs, r1_train_acc, 'b-o', label='Run 1 Train')
ax2.plot(epochs, r1_val_acc,   'b--o', label='Run 1 Val')
ax2.plot(epochs, r2_train_acc, 'r-s', label='Run 2 Train')
ax2.plot(epochs, r2_val_acc,   'r--s', label='Run 2 Val')


ax2.set_title('Accuracy per Epoch - ViT-B-16')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.set_xticks(range(0, 21, 2))
ax2.grid(True)
ax2.legend()

# --- 3. Save ---
plt.tight_layout()
filename = "vit_b_training_results.png"
plt.savefig(filename)
print(f"Graph saved to {filename}")