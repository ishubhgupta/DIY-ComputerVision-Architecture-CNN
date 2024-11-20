# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Developer details: 
# Name: Akshat Rastogi, Shubh Gupta and Rupal Mishra
# Role: Developers
# Code ownership rights: PreProd Corp

# Description: This Streamlit app allows users to input features and make predictions using Neural Network.
# MQs: No
# Cloud: No
# Data versioning: No
# Data masking: No

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
# Environment:     
# Python 3.10.11
# streamlit 1.40.0

import torch

# Function to validate a trained model on a validation dataset
def validate_model(model, val_loader):
    # Set the model to evaluation mode (disables dropout, etc., for deterministic output)
    model.eval()
    
    # Initialize counters for correctly classified samples and total samples
    correct = 0
    total = 0
    
    # Disable gradient computation as it's not needed during validation, improving efficiency
    with torch.no_grad():
        # Iterate through the validation data loader
        for inputs, labels in val_loader:
            # Perform a forward pass with the model on the inputs
            outputs = model(inputs)
            
            # Get the predicted class for each sample (class with the highest score)
            _, predicted = torch.max(outputs, 1)
            
            # Increment the total count of labels (samples) processed
            total += labels.size(0)
            
            # Count correct predictions by comparing predictions with actual labels
            correct += (predicted == labels).sum().item()
    
    # Calculate the accuracy percentage
    accuracy = 100 * correct / total
    return f'Validation Accuracy: {accuracy}%'
