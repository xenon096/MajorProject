import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Text, Scrollbar
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image as PILImage, ImageTk  # For displaying images in Tkinter

# Global variables for the dataset path, model, and class labels
global model, train_dir, class_labels

# Function to upload dataset (Single folder containing four classes)
def load_dataset():
    global train_dir, class_labels
    folder_path = filedialog.askdirectory(title="Select Dataset Folder")
    
    if folder_path:
        train_dir = folder_path  # The entire folder with subfolders for each class
        
        # Check if subfolders (classes) are inside the selected folder
        if os.path.isdir(train_dir):
            subfolders = os.listdir(train_dir)
            class_labels = subfolders  # Store class labels (folder names)
            text.delete('1.0', tk.END)
            text.insert(tk.END, f"Dataset loaded from: {train_dir}\n")
            text.insert(tk.END, f"Classes found: {', '.join(subfolders)}\n")
        else:
            messagebox.showerror("Error", "Please select a valid dataset folder with subfolders for each class.")
    else:
        messagebox.showerror("Error", "Please select a valid dataset folder.")

# Image Preprocessing (Resizing and normalizing)
def preprocess_data():
    global train_dir
    if not train_dir:
        messagebox.showerror("Error", "Please upload a dataset first.")
        return

    # Image Preprocessing
    image_size = (150, 150)  # Resize images to 150x150

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Using 20% data for validation

    # Training data generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=32,
        class_mode='categorical',  # Multi-class classification (4 classes)
        subset='training'  # Use 80% of data for training
    )

    # Validation data generator
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=32,
        class_mode='categorical',  # Multi-class classification (4 classes)
        subset='validation'  # Use 20% of data for validation
    )

    text.delete(1.0, tk.END)
    text.insert(tk.END, "Data Preprocessing Completed.\n")
    text.insert(tk.END, f"Training set: {train_generator.samples} images\n")
    text.insert(tk.END, f"Validation set: {validation_generator.samples} images\n")

# CNN-based model for image classification
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation='softmax'))  # 4 classes
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# VGG16 model for image classification
def create_vgg16_model(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freezing base model layers
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation='softmax'))  # 4 classes
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the selected model
def train_model(model_type):
    global model, train_dir
    if not train_dir:
        messagebox.showerror("Error", "Preprocessing data first is required.")
        return

    # Assuming image size is (150, 150, 3)
    if model_type == 'CNN':
        model = create_cnn_model((150, 150, 3))
    elif model_type == 'VGG16':
        model = create_vgg16_model((150, 150, 3))

    # Get data generators for training and validation
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    history = model.fit(train_generator, epochs=2, validation_data=validation_generator)

    # Show training results
    text.insert(tk.END, f"Model Training Completed.\n")
    text.insert(tk.END, f"Final Accuracy: {history.history['accuracy'][-1]:.4f}\n")

    # Plot training and validation accuracy/loss
    plot_history(history)

# Plot training and validation accuracy/loss
def plot_history(history):
    # Accuracy plot
    plt.figure(figsize=(10, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

import cv2  # OpenCV for showing images

# Function to upload an image and predict its class
def upload_and_predict():
    global model, class_labels
    if model is None:
        messagebox.showerror("Error", "Model is not trained yet. Please train the model first.")
        return
    
    # Select image to upload
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    
    if not file_path:
        return

    # Preprocess image
    img = image.load_img(file_path, target_size=(150, 150))  # Resize image to match model input
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize image

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]  # Get class name from class_labels

    # Display prediction result
    text.delete(1.0, tk.END)
    text.insert(tk.END, f"Predicted Class: {predicted_class}\n")

    # Load the original image for Tkinter display
    img_original = PILImage.open(file_path)
    img_original = img_original.resize((150, 150))  # Resize the image to match the display size
    img_original_tk = ImageTk.PhotoImage(img_original)

    # Display the original image in Tkinter
    original_image_label.config(image=img_original_tk)
    original_image_label.image = img_original_tk  # Keep a reference to the image to prevent garbage collection

    # Now use OpenCV to display the image in a separate window
    img_cv2 = cv2.imread(file_path)  # Read image using OpenCV
    img_cv2 = cv2.resize(img_cv2, (150, 150))  # Resize image to match the model size

    # Display the image using OpenCV (in a separate window)
    cv2.imshow('Predicted Image', img_cv2)

    # Wait for a key press, then close the OpenCV window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Initialize Tkinter Application
root = tk.Tk()
root.title("Alzheimer's Disease Detection Using Deep Learning")
root.geometry("800x700")
root.configure(bg='#f8d7da')  # Light pink background color of the window

# Title label with a different background color
title_label = tk.Label(root, text="Diagnosis of Alzheimer’s Disease Using Convolutional Neural Network With Select Slices by Landmark on Hippocampus in MRI Images", font=("Helvetica", 20, "bold"), bg='skyblue',height=2, width=1)
title_label.pack(pady=10, fill=tk.X)

# Textbox to show output messages with a scrollbar
text_frame = tk.Frame(root, bg='#f8d7da')
text_frame.pack(pady=20)

scrollbar = Scrollbar(text_frame, orient=tk.VERTICAL)
text = Text(text_frame, height=15, width=80, wrap=tk.WORD, font=("Arial", 12, 'bold'), yscrollcommand=scrollbar.set)
scrollbar.config(command=text.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
text.pack(side=tk.LEFT)

# Create a frame for buttons to arrange them in two rows
button_frame = tk.Frame(root, bg='#f8d7da')
button_frame.pack(pady=10)

# First row of buttons
load_button = tk.Button(button_frame, text="Load Dataset", command=load_dataset, height=2, width=15, font=("Arial", 12))
load_button.grid(row=0, column=0, padx=10, pady=5)

preprocess_button = tk.Button(button_frame, text="Preprocess Data", command=preprocess_data, height=2, width=15, font=("Arial", 12))
preprocess_button.grid(row=0, column=1, padx=10, pady=5)

train_cnn_button = tk.Button(button_frame, text="Train CNN Model", command=lambda: train_model('CNN'), height=2, width=15, font=("Arial", 12))
train_cnn_button.grid(row=0, column=2, padx=10, pady=5)

train_vgg_button = tk.Button(button_frame, text="Train VGG16 Model", command=lambda: train_model('VGG16'), height=2, width=15, font=("Arial", 12))
train_vgg_button.grid(row=0, column=3, padx=10, pady=5)

# Second row of buttons
predict_button = tk.Button(button_frame, text="Upload & Predict", command=upload_and_predict, height=2, width=15, font=("Arial", 12))
predict_button.grid(row=1, column=1, padx=10, pady=5)

# Label to display the uploaded image
original_image_label = tk.Label(root, bg='#f8d7da')
original_image_label.pack(pady=10)

# Run the application
root.mainloop()
