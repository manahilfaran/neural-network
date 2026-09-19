import torch

import torch.nn as nn
import torch.optim as optim

from models.neural_network import NeuralNetwork
from data.load_data import load_data


# Load the dataset
train_loader, val_loader, test_loader = load_data()


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

# Track the best validation loss
best_val_loss = float("inf")

# Number of epochs to wait for improvement
patience = 5

# Count epochs without improvement
patience_counter = 0


# Training loop
for epoch in range(epochs):

    # Training mode
    model.train()

    total_train_loss = 0

    for X_batch, y_batch in train_loader:

        # Make predictions
        outputs = model(X_batch).squeeze(1)

        # Calculate the loss
        loss = criterion(outputs, y_batch)

        # Clear previous gradients
        optimizer.zero_grad()

        # Calculate gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = total_train_loss / len(train_loader)


    # Validation mode
    model.eval()

    total_val_loss = 0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            # Make predictions
            outputs = model(X_batch).squeeze(1)

            # Calculate validation loss
            loss = criterion(outputs, y_batch)

            total_val_loss += loss.item()

    average_val_loss = total_val_loss / len(val_loader)


    # Display training and validation loss
    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Training Loss: {average_train_loss:.4f}, "
        f"Validation Loss: {average_val_loss:.4f}"
    )


    # Check whether validation loss improved
    if average_val_loss < best_val_loss:

        # Update the best validation loss
        best_val_loss = average_val_loss

        # Reset the patience counter
        patience_counter = 0

        # Save the best model
        torch.save(
            model.state_dict(),
            "models/neural_network.pth"
        )

        print("Best model saved.")


    else:

        # Increase the patience counter
        patience_counter += 1

        print(
            f"No improvement for "
            f"{patience_counter}/{patience} epochs."
        )


    # Stop training if validation loss has not improved
    if patience_counter >= patience:

        print("Early stopping triggered.")
        break