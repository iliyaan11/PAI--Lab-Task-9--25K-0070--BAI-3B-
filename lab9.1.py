import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

hours = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0]).reshape(-1, 1)
marks = np.array([32, 40, 41, 50, 54, 60, 62, 70, 78, 88])

model = LinearRegression()
model.fit(hours, marks)

slope = model.coef_[0]
intercept = model.intercept_

print(f"Slope (m): {slope:.4f}")
print(f"Intercept (c): {intercept:.4f}")

new_hours = np.array([[5.5], [8.0]])
preds_new = model.predict(new_hours)
print(f"Predicted marks for 5.5 hours: {preds_new[0]:.2f}")
print(f"Predicted marks for 8.0 hours: {preds_new[1]:.2f}")

preds = model.predict(hours)
mae = mean_absolute_error(marks, preds)
mse = mean_squared_error(marks, preds)
r2 = r2_score(marks, preds)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

plt.figure(figsize=(7, 4.5))
plt.scatter(hours, marks, color='blue', s=50, label='Actual Data')
plt.plot(hours, preds, color='red', linewidth=2, label=f'Regression Line (y = {slope:.2f}x + {intercept:.2f})')
plt.title('Study Hours vs. Marks Obtained')
plt.xlabel('Hours Studied')
plt.ylabel('Marks')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()