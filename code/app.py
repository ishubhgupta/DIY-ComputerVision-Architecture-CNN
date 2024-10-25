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
import streamlit as st
import zipfile
from evaluate import validate_model
import torch
from classification import classify
# from evaluate import evaluate_model_with_report
from image_helper import normalize, resize, to_rgb, to_tensor
from train import BirdClassificationCNN, train_model
from load import data_load, load_model
from PIL import Image


st.set_page_config(page_title="Indian Bird Classification", page_icon=":cash:", layout="centered")
st.markdown("<h1 style='text-align: center; color: white;'>Indian Bird Classification</h1>", unsafe_allow_html=True)
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Model Config", "Model Training", "Model Evaluation", "Model Prediction", "Model Flow"])
default_path = "data/Master"

with tab1:
    # Input field to take the directory path
    data_path = st.text_input("Enter the path to the folder containing images", value=default_path)
    extraction_dir = data_path
    if os.path.exists(data_path):
        # List all files in the directory and subdirectories
        file_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(data_path) for f in filenames]
        
        # Count the number of image files
        image_count = sum(1 for file_name in file_list if file_name.lower().endswith(('.jpeg', '.jpg')))
        
        # Display the total number of images found
        st.write(f"Number of images found: {image_count}")
        st.write(f"Images found in: {data_path}")
    else:
        st.write("The specified path does not exist. Please enter a valid path.")


with tab2:
    st.subheader("Model Training")
    st.write("This is where you can train the model.")
    st.divider()

    model_name = 'CNN'
    st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)
    epochs = st.number_input('Number of Epochs:', min_value=1, max_value=100, value=10, step=1)

    if st.button(f"Train {model_name} Model", use_container_width=True):
        with st.status(f"Training {model_name} Model..."):
            train_loader, val_loader = data_load(extraction_dir)
            model, training_report = train_model(train_loader, val_loader, epochs)
            accuracy = validate_model(model, val_loader)
            for report in training_report:
                st.write(f"Training complete! -> {report}")

        st.success(f"{model_name} Trained Sucessully")

        st.write(f"Accuracy: {accuracy}")

    st.divider()


    with tab3:
        st.subheader("Model Evaluation")
        st.write("This is where you can see the current metrics of the latest saved model.")
        st.divider()
        st.markdown(f"<h3 style='text-align: center; color: white;'> Classification Report </h3>", unsafe_allow_html=True)
        try:
            # Loading model
            model_path = r"code\saved_model\bird_classification_cnn.pth"
            st.write(f"Loading model from {model_path}")  # Debug: Print the model path
            model = load_model(model_path)
            model.eval()
            st.write("Model loaded successfully!")  # Debug: Confirm the model is loaded

            # Load the validation data
            st.write("Loading validation data...")  # Debug: Add message before loading data
            train_loader, val_loader = data_load(default_path)
            st.write(f"Validation data loaded. Total batches: {len(val_loader)}")  # Debug: Confirm data loaded

            # Validate the model
            st.write("Validating model...")  # Debug: Add message before validation
            cf = validate_model(model, val_loader)
            st.text(cf)

            st.divider()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    with tab4:
    # Image uploader with improved title and description
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # try:
                # Open and display the uploaded image
                image = Image.open(uploaded_file)
                # Display the predicted class
                predicted_class = classify(uploaded_file)
                st.write(predicted_class)
                # Show the image with an improved caption
                st.image(image, caption="Uploaded Image", use_column_width=True)
            # except Exception as e:
            #     st.error(f"Error processing the image: {e}")
        else:
            st.write("No image uploaded yet. Please upload an image to classify.")

    with tab5:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="second")
        if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                st.divider()
                st.markdown(f"<h4 style='text-align: center; color: white;'>Image Resizing</h4>", unsafe_allow_html=True)
                st.write("Resizing the image ensures that all images have the same dimensions (img_height, img_width) so they can be processed in batches and passed through the network consistently. This also reduces computational complexity when using large images.")
                img = resize(image)
                st.image(img, caption="Resized Image")
                st.divider()
                st.markdown(f"<h4 style='text-align: center; color: white;'> To RGB</h4>", unsafe_allow_html=True)
                st.write("Some images may have different formats (e.g., grayscale, RGBA). The conversion to RGB ensures consistency by transforming all images into 3-channel RGB format, which is necessary if you're using models that expect color images. This step prevents potential errors during processing.")
                img = to_rgb(img)
                st.image(img)
                st.divider()
                st.markdown(f"<h4 style='text-align: center; color: white;'> Convert Image to Tensor</h4>", unsafe_allow_html=True)
                st.write("Machine learning models, especially in PyTorch, expect data in tensor format. This transformation converts the image from a PIL image (or NumPy array) to a PyTorch tensor and scales the pixel values to the range [0, 1], which is better suited for numerical computation.")
                tensor = to_tensor(img)
                st.write(tensor)
                st.image("code/saved_file/tensor.jpg")
                st.divider()
                st.markdown(f"<h4 style='text-align: center; color: white;'> Normalizing</h4>", unsafe_allow_html=True)
                st.write("Normalization adjusts the pixel values in each channel (RGB) to have zero mean and unit variance based on the specific mean and standard deviation. This helps stabilize and speed up training by ensuring that the inputs to the neural network are on a consistent scale, leading to faster convergence. The values [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225] are commonly used for datasets like ImageNet to match pre-trained model expectations.")
                normalized_tensor = normalize(tensor)
                st.write(normalized_tensor)
                st.divider()

                st.write("After that we add a batch dimension to the tensor.")
                batch_normalized_tensor = normalized_tensor.unsqueeze(0)
                st.write(batch_normalized_tensor)

                st.markdown(f"<h3 style='text-align: center; color: white;'>After all this pre-processing the model is passed through the model and classification is done.</h3>", unsafe_allow_html=True)

                st.divider()

                st.markdown(f"<h1 style='text-align: center; color: white;'>Other types of pre-processing techniques.</h1>", unsafe_allow_html=True)




        else:
            st.write("No image uploaded yet. Please upload an image to classify.")




