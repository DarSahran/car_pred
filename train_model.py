# train_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# 1. Create a dummy dataset
data = {
    'age': [5, 3, 8, 2, 1, 7, 4, 6],
    'mileage': [50000, 30000, 80000, 20000, 10000, 70000, 40000, 60000],
    'price': [5000, 7000, 3000, 10000, 12000, 4000, 6000, 3500]
}
df = pd.DataFrame(data)

# 2. Train the model
X = df[['age', 'mileage']]
y = df['price']
model = LinearRegression().fit(X, y)

# 3. Save the model
joblib.dump(model, 'car_price_model.pkl')
print("Model saved successfully!")
