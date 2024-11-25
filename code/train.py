# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Akshat Rastogi, Shubh Gupta and Rupal Mishra
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (26 October 2024)
            # Developers: Akshat Rastogi, Shubh Gupta and Rupal Mishra
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: Implements the CNN model architecture and training pipeline for bird classification.
    # Features:
        # - Custom CNN architecture definition
        # - Model training loop implementation
        # - Model saving functionality


# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # torch 2.5.0
        # torchvision 0.20.0
        # numpy 1.24.3

import torch  # PyTorch for tensor operations and neural networks
import torch.nn as nn  # PyTorch module for neural network layers
import torch.optim as optim  # Module for optimization algorithms
from evaluate import validate_model  # Assuming this is your evaluation function
import os  # For handling file system paths and directories

# Set image dimensions
img_height, img_width = 150, 150  # Image dimensions

# Define a CNN model for bird classification
class BirdClassificationCNN(nn.Module):
    def __init__(self, num_classes=25):
        super(BirdClassificationCNN, self).__init__()
        
        # Define convolutional layers with padding
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Define pooling and flatten layers
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling with 2x2 window
        self.flatten = nn.Flatten()  # Flatten layer output for the fully connected layers
        
        # Define fully connected layers and dropout
        self.fc1 = nn.Linear(128 * (img_height // 16) * (img_width // 16), 512)  # Adjust according to image size
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        self.fc2 = nn.Linear(512, num_classes)  # Final output layer for classification
        
        # Activation functions
        self.relu = nn.ReLU()  # ReLU activation
        self.softmax = nn.Softmax(dim=1)  # Softmax for output layer probabilities

    def forward(self, x):
        # Apply convolutional layers with activation and pooling
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        
        # Flatten the tensor and pass through fully connected layers
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout during training
        x = self.softmax(self.fc2(x))  # Apply softmax to output layer
        
        return x  # Return model predictions

# Function to train the model
def train_model(train_loader, val_loader, epochs, model_save_path):
    # Initialize model, loss function, and optimizer
    model = BirdClassificationCNN(num_classes=25)  # Instantiate the CNN model
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001
    training_report = []  # List to hold training report messages

    # Loop over the number of epochs to train the model
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0  # Initialize running loss for the epoch

        # Loop through batches in the training data loader
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass: compute predictions
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagate gradients
            optimizer.step()  # Update model parameters
            running_loss += loss.item()  # Accumulate batch loss

        # Format and print epoch report
        report = f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}'
        print(report)
        training_report.append(report)  # Store the report

        # Validate the model after each epoch
        validate_model(model, val_loader)  # Call validation function

    # Save model with user-defined path
    model_file = os.path.join(model_save_path, 'bird_classification_cnn.pth')
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    torch.save(model.state_dict(), model_file)
    print(f'Model saved at {model_file}')

    return model, training_report  # Return the trained model and training report

