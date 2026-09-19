import torch
import torch.nn as nn
import torch.optim as optim

from models.neural_network import NeuralNetwork
from data.load_data import load_data


# Load the dataset
train_loader, test_loader = load_data()


# Create the neural network
model = NeuralNetwork(
    input_size=30,
    hidden_sizes=[128, 64, 32],
    output_size=1
)


# Define the loss function
criterion = nn.BCEWithLogitsLoss()


# Define the optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# Number of training epochs
epochs = 50


# Training loop
for epoch in range(epochs):

    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:

        # Make predictions
        outputs = model(X_batch).squeeze(1)

        # Calculating the loss
        loss = criterion(outputs, y_batch)

        # Clearing previous gradients
        optimizer.zero_grad()

        # Calculating gradients
        loss.backward()

        # Updating model parameters
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {average_loss:.4f}"
    )

     


# Saving the trained model
torch.save(model.state_dict(), "models/neural_network.pth")  