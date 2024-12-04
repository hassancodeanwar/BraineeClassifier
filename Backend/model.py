import os
import typing
from typing import Dict, List, Tuple
from functools import partial, reduce
import operator

# Data Handling Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Visualization
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import itertools
from sklearn.metrics import confusion_matrix, classification_report

def compose(*functions):
    """Compose multiple functions."""
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def load_data_paths(base_path: str) -> pd.DataFrame:
    """
    Functional approach to load file paths and labels.
    
    Args:
        base_path (str): Base directory containing image folders
    
    Returns:
        pd.DataFrame: DataFrame with file paths and labels
    """
    def get_file_paths(fold: str) -> List[Tuple[str, str]]:
        fold_path = os.path.join(base_path, fold)
        return [(os.path.join(fold_path, file), fold) 
                for file in os.listdir(fold_path)]
    
    # Get all folders and map to their file paths
    all_files = [item for fold in os.listdir(base_path) 
                 for item in get_file_paths(fold)]
    
    # Convert to DataFrame
    return pd.DataFrame(all_files, columns=['filepaths', 'label'])

def create_data_generators(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    img_size: Tuple[int, int] = (512, 512),
    batch_size: int = 8
) -> Tuple[ImageDataGenerator, ImageDataGenerator, Dict[str, int]]:
    """
    Create data generators with enhanced preprocessing and augmentation.
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Testing dataframe
        img_size (Tuple[int, int]): Target image size
        batch_size (int): Batch size for generators
    
    Returns:
        Tuple of train generator, test generator, and class indices
    """
    # Preprocessing function with normalization
    def preprocess(img):
        img = img / 255.0  # Normalize to [0,1]
        return img
    
    # Enhanced data augmentation for training
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Simple preprocessing for test data
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess
    )
    
    train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        x_col='filepaths', 
        y_col='label', 
        target_size=img_size,
        class_mode='categorical', 
        color_mode='rgb', 
        shuffle=True, 
        batch_size=batch_size
    )
    
    test_generator = test_datagen.flow_from_dataframe(
        test_df, 
        x_col='filepaths', 
        y_col='label', 
        target_size=img_size, 
        class_mode='categorical', 
        color_mode='rgb', 
        shuffle=False, 
        batch_size=batch_size
    )
    
    return train_generator, test_generator, train_generator.class_indices

def build_model(
    input_shape: Tuple[int, int, int], 
    num_classes: int
) -> keras.Model:
    """
    Build and compile a model using VGG19 as the base.
    
    Args:
        input_shape (Tuple[int, int, int]): Input image shape
        num_classes (int): Number of classification classes
    
    Returns:
        keras.Model: Compiled model
    """
    # Base model with transfer learning
    base_model = tf.keras.applications.VGG19(
        include_top=False,  # Exclude fully connected layers at the top
        weights='imagenet',  # Use pre-trained ImageNet weights
        input_shape=input_shape
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)  # Replace Flatten with GAP
    x = layers.BatchNormalization()(x)
    x = layers.Dense(
        1024, 
        activation='relu', 
        kernel_regularizer=regularizers.l2(0.0005)
    )(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(
        512, 
        activation='relu', 
        kernel_regularizer=regularizers.l2(0.0005)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    output = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = keras.Model(inputs=base_model.input, outputs=output)
    
    # Fine-tuning: Unfreeze top layers of the base model
    for layer in base_model.layers[-4:]:  # Unfreeze the last 4 layers
        layer.trainable = True
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(
    model: keras.Model, 
    train_gen, 
    test_gen, 
    epochs: int = 100
) -> Dict[str, List[float]]:
    """
    Train the model with advanced callbacks.
    
    Args:
        model (keras.Model): Compiled model
        train_gen: Training data generator
        test_gen: Validation data generator
        epochs (int): Maximum training epochs
    
    Returns:
        Dict of training history
    """
    # Learning rate reduction callback
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2,  # Reduce learning rate by 80%
        patience=5,  # Wait 5 epochs before reducing
        min_lr=0.00001,  # Minimum learning rate
        verbose=1
    )
    
    # Early stopping with restored best weights
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )
    
    # Training with multiple callbacks
    history = model.fit(
        x=train_gen, 
        epochs=epochs, 
        validation_data=test_gen, 
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return history.history

def evaluate_model(model: keras.Model, test_gen) -> Tuple[float, float]:
    """
    Evaluate model performance.
    
    Args:
        model (keras.Model): Trained model
        test_gen: Test data generator
    
    Returns:
        Tuple of test loss and accuracy
    """
    test_loss, test_acc = model.evaluate(test_gen)
    return test_loss, test_acc





def visualize_data_distribution(df, title='Data Distribution'):
    """
    Create a histogram of class distribution using Plotly.
    
    Args:
        df (pd.DataFrame): DataFrame with label column
        title (str): Plot title
    
    Returns:
        plotly figure object
    """
    fig = px.histogram(
        data_frame=df,
        y='label',
        template='plotly_dark',
        color='label',
        title=title
    )
    fig.show()
    return fig

def visualize_training_images(train_gen, classes, num_images=8):
    """
    Visualize training images with their labels.
    
    Args:
        train_gen: Training data generator
        classes (list): List of class names
        num_images (int): Number of images to display
    """
    images, labels = next(train_gen)
    
    plt.figure(figsize=(20, 20))
    for i in range(min(num_images, len(images))):
        plt.subplot(4, 4, i+1)
        image = images[i] / 255
        plt.imshow(image)
        index = np.argmax(labels[i])
        class_name = classes[index]
        plt.title(class_name, color='blue', fontsize=12)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """
    Plot training and validation loss/accuracy.
    
    Args:
        history (dict): Training history dictionary
    """
    train_acc = history.get('accuracy', [])
    train_loss = history.get('loss', [])
    val_acc = history.get('val_accuracy', [])
    val_loss = history.get('val_loss', [])

    index_loss = np.argmin(val_loss) if val_loss else 0
    index_acc = np.argmax(val_acc) if val_acc else 0

    epochs = list(range(1, len(train_acc) + 1))

    plt.figure(figsize=(20, 8))
    plt.style.use('fivethirtyeight')

    # Loss Subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'g', label='Validation Loss')
    plt.scatter(index_loss + 1, val_loss[index_loss], 
                s=150, c='blue', 
                label=f'Best Loss Epoch = {index_loss + 1}')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy Subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'r', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'g', label='Validation Accuracy')
    plt.scatter(index_acc + 1, val_acc[index_acc], 
                s=150, c='blue', 
                label=f'Best Accuracy Epoch = {index_acc + 1}')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(true_labels, predicted_labels, classes):
    """
    Create and plot confusion matrix.
    
    Args:
        true_labels (array): True class labels
        predicted_labels (array): Predicted class labels
        classes (list): List of class names
    """
    cm = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], 
                 horizontalalignment='center', 
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def print_classification_report(true_labels, predicted_labels, classes):
    """
    Print detailed classification report.
    
    Args:
        true_labels (array): True class labels
        predicted_labels (array): Predicted class labels
        classes (list): List of class names
    
    Returns:
        str: Classification report
    """
    report = classification_report(true_labels, predicted_labels, target_names=classes)
    print(report)
    return report

def visualize_predictions(model, test_gen, classes, num_images=5):
    """
    Visualize model predictions on test images.
    
    Args:
        model: Trained keras model
        test_gen: Test data generator
        classes (list): List of class names
        num_images (int): Number of images to predict and display
    """
    # Get predictions
    predictions = model.predict(test_gen)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get actual images and labels
    test_gen.reset()
    images, true_labels = next(test_gen)
    true_class_indices = np.argmax(true_labels, axis=1)
    
    plt.figure(figsize=(15, 5))
    for i in range(min(num_images, len(images))):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i] / 255.0)
        
        pred_class = classes[predicted_classes[i]]
        true_class = classes[true_class_indices[i]]
        
        color = 'green' if pred_class == true_class else 'red'
        plt.title(f'Pred: {pred_class}\nTrue: {true_class}', color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    print('-'*50)
    print('Start')
    
    # Configuration
    IMG_SIZE = (512, 512)
    BATCH_SIZE = 8

    print('-'*50)
    print('Data Loading')
    # Data Loading
    train_df = load_data_paths('/kaggle/input/brats-2019-traintestvalid/dataset/train')
    test_df = load_data_paths('/kaggle/input/brats-2019-traintestvalid/dataset/valid')
    print('Done')

    print('-'*50)
    print('Create Generators')
    # Create Generators
    train_gen, test_gen, class_indices = create_data_generators(
        train_df, test_df, IMG_SIZE, BATCH_SIZE
    )
    print('Done')
    
    print('-'*50)
    print('Model Setup')
    # Model Setup
    model = build_model(
        input_shape=(*IMG_SIZE, 3), 
        num_classes=len(class_indices)
    )
    print('Done')
    print('-'*50)

    
    print('Training')
    # Training
    history = train_model(model, train_gen, test_gen)
    
    
    print('-'*50)
    # Evaluation
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
    print('Done')

    print('-'*50)
    print('Save Model')
    # Save Model
    model.save('Brain_Tumor_Classifier_Enhanced.h5')
    print('Done')

    print('-'*50)
    print('Visualizations')
    # Visualizations
    print('-'*30)
    print('visualize_data_distribution')
    visualize_data_distribution(train_df)
    print('Done')

    print('-'*30)
    print('visualize_training_images')
    visualize_training_images(train_gen, list(class_indices.keys()))
    print('Done')
    
    print('-'*30)
    print('plot_training_history')
    plot_training_history(history)
    print('Done')

    print('-'*30)
    print('Prediction and Evaluation Visualizations')
    # Prediction and Evaluation Visualizations
    y_pred = model.predict(test_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print('-'*30)
    print('plot_confusion_matrix')
    plot_confusion_matrix(test_gen.classes, y_pred_classes, list(class_indices.keys()))
    print('Done')

    print('-'*30)
    print('print_classification_report')    
    print_classification_report(test_gen.classes, y_pred_classes, list(class_indices.keys()))
    print('Done')

    print('-'*30)
    print('visualize_predictions')
    visualize_predictions(model, test_gen, list(class_indices.keys()))
    print('Done')

    print('-'*50)
    print('All Done')


if __name__ == "__main__":
    main()
