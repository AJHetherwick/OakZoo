"""
This project provides species predictions for Urban Wildlife photos provided by the Oakland Zoo.
Created by Adam Hetherwick with slight guidance from Sammantha Sammons and Darren Minier.
Started 05/27/2025
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress information logs from tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Input
from keras.callbacks import EarlyStopping
from torchvision import transforms, models
from PIL import Image



def main() -> None:

    label_df_path = "C:/Users/accrintern/Documents/AdamH_Project_Files/Oakland_Zoo_Train_Test_Final.csv"
    label_df = pd.read_csv(label_df_path)
    
    train_tensors, train_labels, test_tensors, test_labels = reformat_data(label_df)

    print('train_tensors:', train_tensors.shape,
          'train_labels:', train_labels.shape,
          'test_tensors:', test_tensors.shape,
          'test_labels:', test_labels.shape)

    fit_cnn(train_tensors, train_labels, test_tensors, test_labels)


def reformat_data(label_df: pd.DataFrame) -> tuple[list, list, list, list]:

    # Check the total amount of photos for each species. Put 80% of that into train, 20% into test.
    # Make sure test set does not have augmented photos

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_tensors, train_labels = [], []
    test_tensors, test_labels = [], []

    species_freq = label_df['Label'].value_counts()

    for species, count in species_freq.items():

        train_amount = round(count * 0.8)
        test_amount = count - train_amount

        species_subset = label_df[label_df['Label'] == species]

        # Counter to see how many train and test photos were added.
        train_added, test_added = 0, 0
        index = 0

        while train_added < train_amount or test_added < test_amount:

            row = species_subset.iloc[index]

            img = Image.open(row['File_Path']).convert("RGB")
            img_tensor = transform(img).permute(1, 2, 0)    # (channels, height, width) -> (height, width, channels)

            if train_added < train_amount and row['Group'] == 'train':
                # Add train photo if selected image part of train set

                train_tensors.append(img_tensor)
                train_labels.append(row['Label'])

                train_added += 1

            elif test_added < test_amount and row['Group'] == 'test':
                # Add test photo if selected image part of test set

                test_tensors.append(img_tensor)
                test_labels.append(row['Label'])

                test_added += 1
                
            elif train_added < train_amount and row['Group'] == 'test':
                # Add test photo to train photos ONLY if there are no more train photos

                train_tensors.append(img_tensor)
                train_labels.append(row['Label'])

                train_added += 1
            
            elif test_added < test_amount and row['Group'] == 'train':
                # Add train photo to test photos ONLY if there are no more test photos

                test_tensors.append(img_tensor)
                test_labels.append(row['Label'])

                test_added += 1

            index += 1

    # Convert lists to arrays compatible for tensorflow
    train_tensors = np.array(train_tensors)
    train_labels = np.array(train_labels)
    test_tensors = np.array(test_tensors)
    test_labels = np.array(test_labels)

    # Convert labels from [skunk, raccoon, domestic dog] to [0, 1, 2, ..., 32]
    encoder = LabelEncoder()
    train_labels = encoder.fit_transform(train_labels)
    test_labels = encoder.fit_transform(test_labels)

    return train_tensors, train_labels, test_tensors, test_labels


def fit_cnn(train_tensors: list, train_labels: list, test_tensors: list, test_labels: list) -> None:
    
    # CNN
    cnn = Sequential()

    cnn.add(Input(shape=(224, 224, 3)))
    cnn.add(Conv2D(64,(3,3)))
    cnn.add(LeakyReLU(negative_slope=0.2))   # Leaky Rectified Linear Unit, chooses negative y (slope -0.2) with negative inputs, or y=x if positive
    cnn.add(BatchNormalization())   # Normalizes activations for model speed, performance
    cnn.add(Conv2D(64,(3,3)))   # Fit a 3x3 feature map to scan tensors for important features (fur, lines, clothes, etc)
    cnn.add(LeakyReLU(negative_slope=0.2))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2,2)))  # In the feature maps, select values with strongest features

    cnn.add(Conv2D(64,(3,3)))
    cnn.add(LeakyReLU(negative_slope=0.2))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(64,(3,3)))
    cnn.add(LeakyReLU(negative_slope=0.2))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2,2)))

    cnn.add(Dropout(0.5))   # Drop neurons with little to no predictive ability (a bush over there means skunk)
    cnn.add(Flatten())      # Vectorize
    cnn.add(Dense(128))     # Add first layer of neural network, 128 neurons in length
    cnn.add(LeakyReLU(negative_slope=0.2))
    cnn.add(Dense(64))
    cnn.add(LeakyReLU(negative_slope=0.2))
    cnn.add(Dense(32))
    cnn.add(Dropout(0.5))
    cnn.add(LeakyReLU(negative_slope=0.2))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(33, activation='softmax')) # Add activation function of output layer, softmax compatible with multiclass

    # Compile
    cnn.compile(
        optimizer='adam',   # Adaptive Movement Estimator determines how to update weights
        loss='sparse_categorical_crossentropy',    # Loss function, compatible for multiclass
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
                      validation_data=(test_tensors, test_labels), callbacks=[early_stopping])

    cnn.evaluate(test_tensors, test_labels)


main()
