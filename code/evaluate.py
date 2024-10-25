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

import torch
# from sklearn.metrics import classification_report
# from train import BirdClassificationCNN
# from load import data_load

def validate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return f'Validation Accuracy: {accuracy}%'

# def evaluate_model_with_report(data_entry):
#     all_preds = []
#     all_labels = []
#     model = BirdClassificationCNN(num_classes=25)
#     model.load_state_dict(torch.load('code/saved_model/bird_classification_cnn.pth'))
#     model.eval()

#     _, test_loader = data_load(data_entry)
    
#     with torch.no_grad():
#         for images, labels in test_loader:
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
            
#             all_preds.extend(predicted.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
    
#     # Generate a classification report
    
#     report = classification_report(all_labels, all_preds, target_names=test_loader.classes)
#     return report
