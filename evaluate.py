import torch

from models.neural_network import NeuralNetwork
from data.load_data import load_data


# Loading the test data
train_loader, test_loader = load_data()


# Creating the neural network
model = NeuralNetwork(
    input_size=30,
    hidden_sizes=[128, 64, 32],
    output_size=1
)


# Loading the trained model
model.load_state_dict(
    torch.load("models/neural_network.pth")
)


# Setting the model to evaluation mode
model.eval()


# Tracking correct predictions
correct = 0
total = 0


# Disabling gradient calculation
with torch.no_grad():

    for X_batch, y_batch in test_loader:

        # Make predictions
        outputs = model(X_batch).squeeze(1)

        # Convert logits to probabilities
        probabilities = torch.sigmoid(outputs)

        # Convert probabilities to 0 or 1
        predictions = (probabilities >= 0.5).float()

        # Count correct predictions
        correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)


# Calculating accuracy
accuracy = correct / total

print(f"Test Accuracy: {accuracy:.4f}")