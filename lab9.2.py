import numpy as np
import matplotlib.pyplot as plt

hours = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0])
marks = np.array([32, 40, 41, 50, 54, 60, 62, 70, 78, 88])

def gradient_descent(x, y, lr, epochs):
    m, c = 0.0, 0.0
    n = len(x)
    history = []
    
    for ep in range(1, epochs + 1):
        y_pred = m * x + c
        dm = (-2 / n) * np.sum(x * (y - y_pred))
        dc = (-2 / n) * np.sum(y - y_pred)
        m -= lr * dm
        c -= lr * dc
        
        mse = np.mean((y - y_pred) ** 2)
        history.append(mse)
        
        if ep % 100 == 0:
            print(f"  Epoch {ep:3d} | MSE: {mse:.2f}")
            
    return m, c, history

rates = [0.001, 0.01, 0.05]
epochs = 500
curves = {}

plt.figure(figsize=(8, 4.5))

for lr in rates:
    print(f"\n--- Running GD with learning rate = {lr} ---")
    m, c, losses = gradient_descent(hours, marks, lr, epochs)
    curves[lr] = (m, c, losses)
    plt.plot(range(1, epochs + 1), losses, label=f"lr = {lr} (final m={m:.2f}, c={c:.2f})")

plt.yscale('log')
plt.title('MSE Loss Curve Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()