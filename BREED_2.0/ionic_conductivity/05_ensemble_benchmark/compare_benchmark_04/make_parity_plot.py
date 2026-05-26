

import numpy as np
import matplotlib.pyplot as plt

# Load both
your_y_true = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
your_y_pred = model.predict(X_test)

obelix_y_true = np.load('obelix_y_true.npy')
obelix_y_pred = np.load('obelix_predictions.npy')

your_mae = np.mean(np.abs(your_y_true - your_y_pred))
obelix_mae = np.mean(np.abs(obelix_y_true - obelix_y_pred))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), dpi=150)

# Shared limits for fair comparison
lo = min(your_y_true.min(), your_y_pred.min(), obelix_y_true.min(), obelix_y_pred.min()) - 0.5
hi = max(your_y_true.max(), your_y_pred.max(), obelix_y_true.max(), obelix_y_pred.max()) + 0.5

for ax, yt, yp, mae, title, color in [
    (axes[0], obelix_y_true, obelix_y_pred, obelix_mae, 'OBELiX baseline (RF)', '#888780'),
    (axes[1], your_y_true, your_y_pred, your_mae, 'My GBT (physics-based features)', '#534AB7'),
]:
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5)
    ax.scatter(yt, yp, s=40, alpha=0.6, edgecolor='white', linewidth=0.5, color=color)
    ax.set_xlabel('True log₁₀ σ (S/cm)', fontsize=11)
    ax.set_ylabel('Predicted log₁₀ σ (S/cm)', fontsize=11)
    ax.set_title(f'{title}\nMAE = {mae:.3f}', fontsize=12, pad=10)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Predicting ionic conductivity of solid-state electrolytes', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('parity_comparison.png', dpi=200, bbox_inches='tight')
plt.show()