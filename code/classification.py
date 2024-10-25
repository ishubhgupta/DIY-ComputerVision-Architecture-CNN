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
import os
import torch
from torchvision import models, transforms
from train import BirdClassificationCNN
from ingest_transform import preprocess

import torch
from torchvision import models

def load_model(model_choice):
    if model_choice == "Self-trained CNN":
        # Load self-trained CNN model
        model = BirdClassificationCNN(num_classes=25)
        model.load_state_dict(torch.load('code/saved_model/bird_classification_cnn.pth'))
    elif model_choice == "Pretrained ResNet":
        # Load a pretrained ResNet model
        model = models.resnet18(pretrained=True)  # Use ResNet18 as an example
        
        # Adjust the final layer to match the number of classes
        model.fc = torch.nn.Linear(model.fc.in_features, 25)

        # Load the state dict, allowing for unexpected keys and size mismatches
        pretrained_dict = torch.load('code/saved_model/pretrained_resnet.pth')

        # Filter the state_dict for the ResNet model
        model_dict = model.state_dict()

        # Update the model's state_dict with the pretrained weights, allowing size mismatches
        for key in pretrained_dict.keys():
            if key in model_dict:
                if pretrained_dict[key].size() == model_dict[key].size():
                    model_dict[key] = pretrained_dict[key]

        model.load_state_dict(model_dict)  # Load the updated state dict

    model.eval()
    return model


def predict_image_class(model, img_tensor, class_labels):
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_labels[predicted.item()]
        return predicted_class

def classify(image, model_choice):
    model = load_model(model_choice)  # Load the selected model

    # Preprocess the image
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    data_dir = "data/Master"
    class_names = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]

    # Predict the class
    predicted_class = predict_image_class(model, image_tensor, class_names)
    return predicted_class
