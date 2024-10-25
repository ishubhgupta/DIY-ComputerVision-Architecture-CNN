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
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import psycopg2
import streamlit as st


img_height, img_width = 150, 150 

def preprocess(img_path):
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path)
    img = img.convert("RGB")  # Ensure the image is in RGB format
    img = transform(img)
    return img


# Function to connect to PostgreSQL database
import psycopg2
import streamlit as st

# PostgreSQL connection function
def connect_postgresql():
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="indian_bird",
            user="postgres",
            password="123456"
        )
        return conn
    except Exception as e:
        st.write(f"PostgreSQL connection error: {e}")
        return None

# Function to store data path in PostgreSQL
def store_data_path_in_postgresql(data_path):
    conn = connect_postgresql()
    if conn is not None:
        try:
            conn.autocommit = True
            cur = conn.cursor()
            
            # Create table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS data_paths (
                    id SERIAL PRIMARY KEY,
                    path TEXT NOT NULL
                )
            """)
            
            # Insert the data path
            cur.execute("INSERT INTO data_paths (path) VALUES (%s)", (data_path,))
            conn.commit()
            st.write("Data path stored in PostgreSQL successfully.")
        except Exception as e:
            st.write(f"Error storing data path in PostgreSQL: {e}")
        finally:
            cur.close()
            conn.close()

# Function to store model path in PostgreSQL
def store_model_path_in_postgresql(model_path):
    conn = connect_postgresql()
    if conn is not None:
        try:
            conn.autocommit = True
            cur = conn.cursor()
            
            # Create model_paths table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_paths (
                    id SERIAL PRIMARY KEY,
                    path TEXT NOT NULL
                )
            """)
            
            # Insert the model path
            cur.execute("INSERT INTO model_paths (path) VALUES (%s)", (model_path,))
            conn.commit()
            st.write("Model path stored in PostgreSQL successfully.")
        except Exception as e:
            st.write(f"Error storing model path in PostgreSQL: {e}")
        finally:
            cur.close()
            conn.close()

import psycopg2
import streamlit as st

def retrieve_data_path_from_postgresql():
    conn = connect_postgresql()
    if conn is not None:
        try:
            cur = conn.cursor()
            # Retrieve the latest data path from PostgreSQL
            cur.execute("SELECT path FROM data_paths ORDER BY id DESC LIMIT 1")
            result = cur.fetchone()
            if result:
                data_path = result[0]
                st.write(f"Retrieved data path from PostgreSQL: {data_path}")
                return data_path
            else:
                st.write("No data path found in PostgreSQL.")
                return None
        except Exception as e:
            st.write(f"Error retrieving data path from PostgreSQL: {e}")
            return None
        finally:
            cur.close()
            conn.close()
