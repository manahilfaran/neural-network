import torch

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from models.neural_network import NeuralNetwork
from data.load_data import load_data


# Load the dataset
train_loader, val_loader, test_loader = load_data()


# Creating the neural network
model = NeuralNetwork(
    input_size=30,
    hidden_sizes=[128, 64, 32],
    output_size=1
)


# Loading the best trained model
model.load_state_dict(
    torch.load("models/neural_network.pth")
)


# Setting the model to evaluation mode
model.eval()


# Tracking correct predictions
correct = 0
total = 0


# Storing predictions and actual labels
all_predictions = []
all_labels = []


# Disabling gradient calculation
with torch.no_grad():

    for X_batch, y_batch in test_loader:

        # Make predictions
        outputs = model(X_batch).squeeze(1)

        # Convert logits to probabilities
        probabilities = torch.sigmoid(outputs)

        # Convert probabilities to 0 or 1
        predictions = (probabilities >= 0.5).float()

        # Store predictions and actual labels
        all_predictions.extend(predictions.numpy())
        all_labels.extend(y_batch.numpy())

        # Count correct predictions
        correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)


# Calculate accuracy
accuracy = correct / total


# Calculate precision, recall, and F1-score
precision = precision_score(
    all_labels,
    all_predictions
)

recall = recall_score(
    all_labels,
    all_predictions
)

f1 = f1_score(
    all_labels,
    all_predictions
)


# Calculate confusion matrix
conf_matrix = confusion_matrix(
    all_labels,
    all_predictions
)


# Display results
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("Confusion Matrix:")
print(conf_matrix)