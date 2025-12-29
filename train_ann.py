import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# read csv
data = pd.read_csv('Mobile_Price_Classification-220531-204702.csv')
print(data.head())
