from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Loading the dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("X shape:", X.shape)
print("y shape:", y.shape)

print("First 5 rows:")
print(X[:5])

print("First 5 targets:")
print(y[:5])

print("Feature names:")
print(data.feature_names)

# Standardizing features using their mean and standard deviation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaled X shape:", X_scaled.shape)
print("First 5 scaled rows:")
print(X_scaled[:5])