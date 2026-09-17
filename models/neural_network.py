import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        layers = []
        previous_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))

            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)