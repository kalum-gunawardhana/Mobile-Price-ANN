import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# load dataset
data = pd.read_csv("Mobile_Price_Classification-220531-204702.csv")

x = data.drop("price_range", axis=1)

scaler = StandardScaler()
scaler.fit(x)
