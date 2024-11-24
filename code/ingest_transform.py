# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Shubh Gupta
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (26 October 2024)
            # Developers: Shubh Gupta
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: Handles PostgreSQL database operations and image preprocessing.
    # Features:
        # - PostgreSQL database connection and operations
        # - Image preprocessing pipeline
        # - Data path storage and retrieval
        # Database Integration:
            # Postgres: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # pillow 10.0.0
        # torch 2.5.0
        # torchvision 0.20.0

# Import necessary libraries
import torch  # PyTorch for model handling
import torch.nn as nn  # Neural network module in PyTorch
from torchvision import transforms  # For preprocessing image data
from PIL import Image  # Image processing
import streamlit as st  # Web app framework

# Define image dimensions for preprocessing
img_height, img_width = 150, 150 

# Preprocess function to apply transformations to an input image
def preprocess(img_path):
    """
    Preprocess the input image by resizing, normalizing, and converting it to a tensor format.
    
    Parameters:
    - img_path (str): The file path to the image.

    Returns:
    - img (Tensor): The transformed image tensor ready for model input.
    """
    # Define the sequence of transformations: resize, convert to tensor, normalize
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),  # Resize image to fixed dimensions
        transforms.ToTensor(),  # Convert image to tensor format
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet means/stds
    ])
    
    # Open the image using PIL, convert it to RGB (in case it is grayscale), and apply transformations
    img = Image.open(img_path)
    img = img.convert("RGB")  # Ensure image is in RGB format for model compatibility
    img = transform(img)
    return img
