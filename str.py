import streamlit as st
import os
import numpy as np
import zipfile
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Initialize session state for trained model
if "trained_model" not in st.session_state:
    st.session_state["trained_model"] = None

global model, train_dir, class_labels
train_dir = "./extracted_dataset"
class_labels = []

@st.cache_resource
def get_model(model_type, input_shape, num_classes):
    if model_type == 'CNN':
        return create_cnn_model(input_shape, num_classes)
    else:
        return create_vgg16_model(input_shape, num_classes)

def load_dataset():
    global train_dir, class_labels
    uploaded_file = st.file_uploader("Upload a ZIP file containing the dataset", type=["zip"])
    if uploaded_file is not None:
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.makedirs(train_dir, exist_ok=True)
        
        zip_path = os.path.join(train_dir, "dataset.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(train_dir)
        os.remove(zip_path)
        
        # Check if extracted dataset is nested inside a single folder
        subfolders = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        if len(subfolders) == 1:
            train_dir = os.path.join(train_dir, subfolders[0])
        
        # Identify subfolders (classes) inside the extracted dataset
        class_labels = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        
        if class_labels:
            st.success("Dataset extracted successfully!")
            st.write(f"Classes found: {', '.join(class_labels)}")
        else:
            st.error("No class subfolders found. Please ensure the dataset is structured correctly.")
    else:
        st.error("Please upload a valid ZIP file.")

def preprocess_data():
    global train_dir, class_labels
    if not os.path.exists(train_dir) or not class_labels:
        st.error("Please upload a dataset first.")
        return None, None

    image_size = (150, 150)
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=image_size, batch_size=32, class_mode='categorical', subset='training')
    
    validation_generator = train_datagen.flow_from_directory(
        train_dir, target_size=image_size, batch_size=32, class_mode='categorical', subset='validation')
    
    class_labels = list(train_generator.class_indices.keys())
    
    st.success("Data Preprocessing Completed.")
    st.write(f"Training set: {train_generator.samples} images")
    st.write(f"Validation set: {validation_generator.samples} images")
    return train_generator, validation_generator

def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_vgg16_model(input_shape, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model_type):
    global train_dir, class_labels
    if not os.path.exists(train_dir) or not class_labels:
        st.error("Please preprocess data first.")
        return
    
    train_generator, validation_generator = preprocess_data()
    if train_generator is None or validation_generator is None:
        return
    
    num_classes = len(class_labels)
    st.session_state["trained_model"] = get_model(model_type, (150, 150, 3), num_classes)
    
    history = st.session_state["trained_model"].fit(train_generator, epochs=2, validation_data=validation_generator)
    st.success("Model Training Completed.")
    st.write(f"Final Accuracy: {history.history['accuracy'][-1]:.4f}")
    plot_history(history)

def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(history.history['accuracy'], label='train accuracy')
    ax[0].plot(history.history['val_accuracy'], label='val accuracy')
    ax[0].set_title('Accuracy')
    ax[0].legend()
    
    ax[1].plot(history.history['loss'], label='train loss')
    ax[1].plot(history.history['val_loss'], label='val loss')
    ax[1].set_title('Loss')
    ax[1].legend()
    
    st.pyplot(fig)

def upload_and_predict():
    if st.session_state["trained_model"] is None:
        st.error("Model is not trained yet. Please train the model first.")
        return
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = st.session_state["trained_model"].predict(img_array)
        predicted_class = st.session_state["class_labels"][np.argmax(prediction)]
        
        st.success(f"Predicted Class: {predicted_class}")

st.title("Alzheimer's Disease Detection Using Deep Learning")

load_dataset()

if st.button("Preprocess Data"):
    preprocess_data()

if st.button("Train CNN Model"):
    train_model('CNN')

if st.button("Train VGG16 Model"):
    train_model('VGG16')

if st.button("Upload & Predict"):
    upload_and_predict()
