# ingest_transform_couchdb.py
import couchdb
import streamlit as st

import couchdb
import streamlit as st

# Function to connect to CouchDB and get the database instance
def connect_couchdb(db_name="data_paths"):
    try:
        couch = couchdb.Server("http://admin:123456@localhost:5984/")
        
        # Check if the database exists; create it if it doesn't
        if db_name not in couch:
            db = couch.create(db_name)
        else:
            db = couch[db_name]
        return db
    except Exception as e:
        st.write(f"CouchDB connection error: {e}")
        return None

# Function to store data path in CouchDB
def store_data_path_in_couchdb(data_path):
    db = connect_couchdb("data_paths")
    if db is not None:
        try:
            # Save the data path document
            db.save({"type": "data_path", "path": data_path})
            st.write("Data path stored in CouchDB successfully.")
        except Exception as e:
            st.write(f"Error storing data path in CouchDB: {e}")

# Function to store model path in CouchDB
def store_model_path_in_couchdb(model_path):
    db = connect_couchdb("model_paths")
    if db is not None:
        try:
            # Save the model path document
            db.save({"type": "model_path", "path": model_path})
            st.write("Model path stored in CouchDB successfully.")
        except Exception as e:
            st.write(f"Error storing model path in CouchDB: {e}")

import couchdb

def retrieve_data_path_from_couchdb():
    db = connect_couchdb("data_paths")
    if db is not None:
        try:
            # Retrieve the last data path document (most recent path stored)
            docs = list(db.find({"selector": {"type": "data_path"}}))
            if docs:
                latest_doc = sorted(docs, key=lambda d: d["_id"], reverse=True)[0]
                data_path = latest_doc.get("path", "")
                st.write(f"Retrieved data path from CouchDB: {data_path}")
                return data_path
            else:
                st.write("No data path found in CouchDB.")
                return None
        except Exception as e:
            st.write(f"Error retrieving data path from CouchDB: {e}")
            return None
