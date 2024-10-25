# load_resnet.py

import torch
import torchvision.models as models
import torch.nn as nn

def create_pretrained_resnet(num_classes):
    # Load the pretrained ResNet model (you can choose ResNet18 or ResNet50)
    model = models.resnet50(pretrained=True)

    # Modify the final fully connected layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Set the model to evaluation mode
    model.eval()

    # Save the model's state dictionary
    torch.save(model.state_dict(), 'code/saved_model/pretrained_resnet.pth')

    print("Pretrained ResNet model saved successfully.")

    return model

