# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Akshat Rastogi, Shubh Gupta and Rupal Mishra
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (19 September 2024)
            # Developers: Akshat Rastogi, Shubh Gupta and Rupal Mishra
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This Streamlit app allows users to input features and make predictions using Unsupervised Learning.
        # SQLite: Yes
        # MQs: No
        # Cloud: No
        # Data versioning: No
        # Data masking: No

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.11.5
        # Streamlit 1.36.0
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

def resize(img):
    resize_transform = transforms.Resize((150, 150))
    img = resize_transform(img)
    return img


def to_tensor(img, show_image=True):
    # Convert the image to RGB format if it's not already in RGB
    img = img.convert("RGB")
    
    # Convert the image to a tensor
    to_tensor_transform = transforms.ToTensor()
    img_tensor = to_tensor_transform(img)
    
    # Optionally display the image
    if show_image:
        # Permute the tensor from [C, H, W] to [H, W, C] for displaying
        np_img = img_tensor.permute(1, 2, 0).numpy()
        plt.imshow(np_img)
        plt.axis('off')  # Remove axes for cleaner visualization
        plt.savefig("code/saved_file/tensor.jpg")
    
    return img_tensor

def normalize(img):
    normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = normalize_transform(img)
    return img

def to_rgb(img):
    img = img.convert("RGB")
    return img


