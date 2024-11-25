# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Akshat Rastogi, Shubh Gupta and Rupal Mishra
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (26 October 2024)
            # Developers: Akshat Rastogi, Shubh Gupta and Rupal Mishra
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: Handles the loading and modification of pretrained ResNet models.
    # Features:
        # - Pretrained ResNet model loading
        # - Model architecture modification for custom classes
        # - Model state saving

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # torch 2.5.0
        # torchvision 0.20.0

import torch  # PyTorch library
import torchvision.models as models  # Pretrained model library in torchvision
import torch.nn as nn  # Neural network module in PyTorch
import os  # Operating system library for file path operations

# Function to create a pretrained ResNet model and adjust its output layer for specific classes
def create_pretrained_resnet(num_classes, model_save_path):
    """
    Load a pretrained ResNet model, modify its final layer for the specified number of classes, 
    set it to evaluation mode, and save its state dictionary.
    
    Parameters:
    - num_classes (int): The number of output classes for the custom dataset.
    - model_save_path (str): The directory path where the model state dictionary will be saved.
    
    Returns:
    - model (torchvision.models.ResNet): Modified ResNet model.
    """
    
    # Load the pretrained ResNet model; ResNet50 is used here (ResNet18 could also be used)
    model = models.resnet50(pretrained=True)  # Set pretrained=True to load ImageNet weights

    # Modify the final fully connected layer to output the specified number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Change to match num_classes
    
    # Set the model to evaluation mode, which is generally used during inference
    model.eval()

    # Save with user-defined path
    save_path = os.path.join(model_save_path, 'pretrained_resnet.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Pretrained ResNet model saved at {save_path}")

    return model  # Return the modified model
