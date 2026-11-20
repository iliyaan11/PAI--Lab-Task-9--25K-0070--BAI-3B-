import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.DataFrame({
    'experience': [1, 2, 3, None, 5, 6, 7, 8, 9, 10, 3, 5],
    'education': ['BS', 'BS', 'MS', 'BS', 'MS', 'PhD', 'MS', 'PhD', 'BS', 'PhD', 'BS', 'MS'],
    'salary_k': [45, 50, 70, 58, 85, 120, 95, 140, 90, 160, 55, 88]
})

df['experience'] = df['experience'].fillna(df['experience'].median())
df = pd.get_dummies(df, columns=['education'], dtype=int)

X = df.drop(columns=['salary_k'])
y = df['salary_k']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

test_preds = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, test_preds)
r2 = r2_score(y_test, test_preds)

print(f"Test MAE: {mae:.2f}")
print(f"Test R² Score: {r2:.4f}")

new_cand = pd.DataFrame([[4, 0, 1, 0]], columns=X.columns)
new_cand_scaled = scaler.transform(new_cand)
pred_salary = model.predict(new_cand_scaled)[0]

print(f"Predicted salary for MS with 4 years experience: {pred_salary:.2f}k")