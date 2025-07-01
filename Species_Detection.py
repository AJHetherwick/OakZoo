"""
This project provides species predictions for Urban Wildlife photos provided by the Oakland Zoo.
Created by Adam Hetherwick with slight guidance from Sammantha Sammons and Darren Minier.
Started 05/27/2025
"""

import pandas as pd
import numpy as np
import matplotlib
import sklearn
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU
from keras.callbacks import EarlyStopping
from torchvision import transforms, models
from PIL import Image
import os



def main() -> None:

    label_df_compressed_path = "C:/Users/accrintern/Documents/AdamH_Project_Files/Oakland_Zoo_Train_Labels.csv"
    label_df_compressed = pd.read_csv(label_df_compressed_path)
    
    train_tensors, train_labels = reformat_data(label_df_compressed)

    fit_cnn(train_tensors, train_labels)


def reformat_data(label_df_compressed) -> tuple[list, list]:

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    tensors = []
    labels = []

    for _, row in label_df_compressed.iterrows():
        img = Image.open(row['File_Path']).convert("RGB")
        img_tensor = transform(img)

        tensors.append(img_tensor)
        labels.append(row['Label_ID'])

    return tensors, labels


def fit_cnn(train_tensors: list, train_labels: list) -> None:
    # CNN
    cnn = Sequential()

    cnn.add(Conv2D(64,(3,3), input_shape=(244, 244, 3)))
    cnn.add(LeakyReLU(alpha=0.2))   # Leaky Rectified Linear Unit, chooses negative y (slope -0.2) with negative inputs, or y=x if positive
    cnn.add(BatchNormalization())   # Normalizes activations for model speed, performance
    cnn.add(Conv2D(64,(3,3)))   # Fit a 3x3 feature map to scan tensors for important features (fur, lines, clothes, etc)
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2,2)))  # In the feature maps, select values with strongest features

    cnn.add(Conv2D(64,(3,3)))
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(64,(3,3)))
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2,2)))

    cnn.add(Dropout(0.5))   # Drop neurons with little to no predictive ability (a bush over there means skunk)
    cnn.add(Flatten())      # Vectorize
    cnn.add(Dense(128))     # Add first layer of neural network, 128 neurons in length
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dense(64))
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dense(32))
    cnn.add(Dropout(0.5))
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(1, activation='softmax')) # Add activation function of output layer, softmax compatible with multiclass

    # Compile
    cnn.compile(
        optimizer='adam',   # Adaptive Movement Estimator determines how to update weights
        loss='categorical_crossentropy',    # Loss function, compatible for multiclass
        metrics=['accuracy'])   # Print and track accuracy

    early_stopping = EarlyStopping(
        monitor='val_accuracy',     # After each epoch iteration (through whole CNN), check val_accuracy
        patience=10,    # If val_accuracy doesn't improve for 10 epochs, early stop
        verbose=1,      # Prints a message when early stopping is triggerered
        restore_best_weights=True)      # Reload best weights after early stop

    # Train CNN
    history = cnn.fit(train_tensors, 
                      train_labels, 
                      epochs=30,    # 30 iterations through CNN maximum (early stop may trigger)
                      verbose=1,    # Track progress bar
                      validation_data=(test_images, test_labels), callbacks=[early_stopping])


main()
