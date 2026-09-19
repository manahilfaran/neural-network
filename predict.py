import torch
import joblib

from models.neural_network import NeuralNetwork


# Load the scaler used during training
scaler = joblib.load("models/scaler.pkl")


# Create the neural network
model = NeuralNetwork(
    input_size=30,
    hidden_sizes=[128, 64, 32],
    output_size=1
)


# Load the trained model
model.load_state_dict(
    torch.load("models/neural_network.pth")
)


# Set the model to evaluation mode
model.eval()


print("Enter the 30 feature values:")

features = []

for i in range(30):
    value = float(input(f"Feature {i + 1}: "))
    features.append(value)

# Scale the input using the same scaler used during training
scaled_features = scaler.transform([features])


# Converting the scaled input into a PyTorch tensor
input_tensor = torch.tensor(
    scaled_features,
    dtype=torch.float32
)


# Making the prediction
with torch.no_grad():

    output = model(input_tensor)

    probability = torch.sigmoid(output).item()


# Converting probability into a class
prediction = 1 if probability >= 0.5 else 0


# Displaying the result
print()
print(f"Predicted Class: {prediction}")
print(f"Probability: {probability:.4f}")