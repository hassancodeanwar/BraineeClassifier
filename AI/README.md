# Brain Tumor Classification Model

## 1. Overview
### Purpose
This script implements a deep learning model for brain tumor classification using transfer learning and convolutional neural networks (CNNs).

### Key Components
- Data Loading
- Image Preprocessing
- Model Architecture
- Training
- Evaluation
- Visualization

## 2. Dependencies
### Required Libraries
- Python Standard Libraries: `os`, `typing`
- Data Manipulation: 
  - `numpy`
  - `pandas`
- Deep Learning:
  - `tensorflow`
  - `keras`
- Visualization:
  - `matplotlib`
  - `plotly`
  - `seaborn`
- Machine Learning Metrics:
  - `sklearn`

## 3. Data Handling

### 3.1 `load_data_paths(base_path: str) -> pd.DataFrame`
**Function**: Recursively load image file paths and their corresponding labels from a directory structure.

**Parameters**:
- `base_path`: Root directory containing image folders

**Returns**:
- DataFrame with columns:
  - `filepaths`: Full path to each image
  - `label`: Corresponding class label

**Approach**:
- Uses functional programming to extract file paths
- Supports flexible directory structures
- Creates a pandas DataFrame for easy manipulation

### 3.2 `create_data_generators()`
**Purpose**: Create data generators for training and testing

**Key Features**:
- Configurable image size (default: 512x512)
- Configurable batch size (default: 8)
- Supports RGB color mode
- Shuffles training data
- Generates categorical labels

## 4. Model Architecture

### 4.1 Base Model: Transfer Learning
- Uses EfficientNetB3 as the base model
- Pre-trained weights from ImageNet
- Max pooling for feature extraction

### 4.2 Custom Layers
- Dense layers with regularization:
  1. 1024 neurons
  2. 512 neurons
  3. 256 neurons

**Regularization Techniques**:
- L1 regularization: 0.0005
- L2 regularization: 0.001
- Dropout: 30%
- Batch Normalization

### 4.3 Compilation Parameters
- Optimizer: Adamax
- Learning Rate: 0.0001
- Loss Function: Categorical Cross-Entropy
- Metrics: Accuracy

## 5. Training Process

### 5.1 Early Stopping
**Configuration**:
- Monitor: Validation Accuracy
- Patience: 10 epochs
- Baseline: 0.999
- Restore Best Weights: Enabled

### 5.2 Training Callback
- Maximum Epochs: 50
- Verbose Logging

## 6. Evaluation Metrics

### 6.1 Quantitative Metrics
- Test Loss
- Test Accuracy
- Confusion Matrix
- Classification Report
  - Precision
  - Recall
  - F1-Score

### 6.2 Visualization Functions
1. Data Distribution
2. Training Images
3. Training History (Loss/Accuracy)
4. Confusion Matrix
5. Prediction Visualizations

## 7. Model Saving
- Saves trained model as 'Brain_Tumor_Classifier02.h5'
- Includes all learned weights and model configuration

## 8. Workflow

```
1. Data Loading
↓
2. Data Preprocessing
↓
3. Model Architecture Construction
↓
4. Model Training
↓
5. Model Evaluation
↓
6. Visualization
↓
7. Model Saving
```

## 9. Hyperparameters and Configurations

### Image Processing
- Size: 512x512 pixels
- Color Mode: RGB
- Batch Size: 8

### Model
- Base Model: EfficientNetB3
- Dense Layers: [1024, 512, 256]
- Regularization: L1 (0.0005), L2 (0.001)
- Dropout: 30%

### Training
- Optimizer: Adamax
- Learning Rate: 0.0001
- Max Epochs: 50
- Early Stopping Patience: 10

## 10. Potential Limitations
- Requires significant computational resources
- Performance depends on dataset quality
- May require domain-specific fine-tuning

## 11. Recommendations for Improvement
1. Implement k-fold cross-validation
2. Experiment with data augmentation
3. Try different base models
4. Add learning rate scheduling
5. Implement more robust error handling

## 12. Usage Instructions
1. Prepare dataset in specified directory structure
2. Adjust hyperparameters as needed
3. Run the script
4. Review model performance and visualizations
5. Save and deploy the model
---


--------------------------------------------------
Start
--------------------------------------------------
Data Loading
Done
--------------------------------------------------
Create Generators
Found 2800 validated image filenames belonging to 2 classes.
Found 200 validated image filenames belonging to 2 classes.
Done
--------------------------------------------------
Model Setup
Done
--------------------------------------------------
Training
Epoch 1/50

2024-12-04 00:58:06.619246: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:961] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inmodel_1/block1b_drop/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer

350/350 [==============================] - 289s 704ms/step - loss: 3.2245 - accuracy: 0.7064 - val_loss: 3.2428 - val_accuracy: 0.7500
Epoch 2/50
350/350 [==============================] - 243s 694ms/step - loss: 2.9995 - accuracy: 0.8389 - val_loss: 3.0133 - val_accuracy: 0.8750
Epoch 3/50
350/350 [==============================] - 242s 692ms/step - loss: 2.8768 - accuracy: 0.8846 - val_loss: 2.9645 - val_accuracy: 0.9050
Epoch 4/50
350/350 [==============================] - 243s 694ms/step - loss: 2.8095 - accuracy: 0.9054 - val_loss: 2.8332 - val_accuracy: 0.9450
Epoch 5/50
350/350 [==============================] - 242s 693ms/step - loss: 2.7814 - accuracy: 0.9143 - val_loss: 2.7944 - val_accuracy: 0.9400
Epoch 6/50
350/350 [==============================] - 242s 692ms/step - loss: 2.7315 - accuracy: 0.9311 - val_loss: 2.7835 - val_accuracy: 0.9400
Epoch 7/50
350/350 [==============================] - 243s 693ms/step - loss: 2.7138 - accuracy: 0.9339 - val_loss: 2.7592 - val_accuracy: 0.9550
Epoch 8/50
350/350 [==============================] - 242s 692ms/step - loss: 2.6474 - accuracy: 0.9607 - val_loss: 2.7226 - val_accuracy: 0.9550
Epoch 9/50
350/350 [==============================] - 243s 693ms/step - loss: 2.6374 - accuracy: 0.9596 - val_loss: 2.6770 - val_accuracy: 0.9650
Epoch 10/50
350/350 [==============================] - 243s 694ms/step - loss: 2.6064 - accuracy: 0.9614 - val_loss: 2.6899 - val_accuracy: 0.9600
--------------------------------------------------
25/25 [==============================] - 4s 146ms/step - loss: 2.6770 - accuracy: 0.9650
Test Loss: 2.677032470703125, Test Accuracy: 0.9649999737739563
Done
--------------------------------------------------
Save Model
Done
--------------------------------------------------
Visualizations
------------------------------
visualize_data_distribution

Done
------------------------------
visualize_training_images


Done
------------------------------
plot_training_history


Done
------------------------------
Prediction and Evaluation Visualizations
25/25 [==============================] - 7s 142ms/step
------------------------------
plot_confusion_matrix


Done
------------------------------
print_classification_report
              precision    recall  f1-score   support

          no       0.93      1.00      0.97       100
         yes       1.00      0.93      0.96       100

    accuracy                           0.96       200
   macro avg       0.97      0.97      0.96       200
weighted avg       0.97      0.96      0.96       200

Done
------------------------------
visualize_predictions
25/25 [==============================] - 4s 142ms/step


Done
--------------------------------------------------
All Done


