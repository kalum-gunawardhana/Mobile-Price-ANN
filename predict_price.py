import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset (ONLY to fit scaler)
data = pd.read_csv("Mobile_Price_Classification-220531-204702.csv")

X = data.drop("price_range", axis=1)

scaler = StandardScaler()
scaler.fit(X)

# Build same ANN model
model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Load trained weights
model.load_weights("mobile_price_ann.weights.h5")

# New mobile phone data (example)
# Order must match dataset columns exactly
new_mobile = np.array([[8, 128, 4, 1, 0, 1, 0, 1, 1, 0]])

# Scale input
new_mobile_scaled = scaler.transform(new_mobile)

# Predict
prediction = model.predict(new_mobile_scaled)

# Result
if prediction[0][0] >= 0.5:
    print("Predicted Price Range: HIGH (1)")
else:
    print("Predicted Price Range: LOW (0)")
