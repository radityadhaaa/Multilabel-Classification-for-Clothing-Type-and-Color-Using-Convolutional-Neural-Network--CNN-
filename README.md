
# **Multilabel Classification for Clothing Type and Color Using CNN**

## Overview
This project involves building a multilabel classification model using a Convolutional Neural Network (CNN) to identify the type (e.g., t-shirt or hoodie) and color (e.g., red, yellow, blue, black, or white) of clothing from images. The project is inspired by the need for automated classification in e-commerce, where efficient inventory management and enhanced customer experience can be achieved through automated image-based categorization. The model was developed and tested using a dataset of labeled clothing images.

## Objectives
The primary goal of this project is to:
1. Train a CNN model to classify clothing images based on type and color.
2. Achieve high accuracy in both training and testing data, demonstrating the model's ability to generalize well.
3. Deploy the model to assist in real-world applications, such as inventory management for online retailers.

## Project Structure
The project is structured as follows:

- **Data Loading and Preprocessing**: Images are loaded, resized, and normalized. Labels for clothing type and color are one-hot encoded to support multilabel classification.
- **Model Building**: A CNN model is created with a shared feature extraction layer and separate output layers for clothing type and color classification.
- **Training and Evaluation**: The model is trained on labeled images and evaluated on both training and testing datasets to assess its performance.
- **Performance Metrics**: Accuracy metrics and confusion matrices are displayed for each label type (type and color) on both training and test data.
- **Submission Preparation**: The model generates predictions on the test set, and results are saved in a CSV file for further use.

## Key Components
1. **Dataset Preparation**:
    - `load_images()` function is used to load images from specified directories, resize them to 128x128, and normalize them.
    - The `train.csv` file is used to map image IDs to their respective labels for training purposes.
   
2. **Model Architecture**:
    - The CNN model includes three convolutional layers followed by max-pooling and flattening layers.
    - Separate fully connected output layers are defined for the two labels: type and color, each using `softmax` activation for classification.

3. **Training**:
    - The model is compiled using the `adam` optimizer and `categorical_crossentropy` loss function for each output.
    - Model accuracy is visualized through accuracy plots over training epochs.

4. **Evaluation and Metrics**:
    - Confusion matrices and classification reports are generated for both training and test data to provide insight into model performance.
    - The model performance is visualized using confusion matrices for each label type (type and color).

5. **Prediction and Submission**:
    - Predictions are generated for the test dataset and saved to `test_labels.csv` for further evaluation and integration.

## Results
The model achieved high accuracy on both training and testing datasets, suggesting strong generalization. Evaluation metrics include confusion matrices and classification reports, indicating reliable performance in real-world applications.

## Requirements
To run this project, the following libraries are required:
- `numpy`
- `matplotlib`
- `tensorflow`
- `pandas`
- `Pillow`
- `scikit-learn`

These libraries can be installed using `pip install` if they are not already available in your environment.

## Usage
1. **Data Preparation**:
   - Update paths to training and test directories in the code.
   - Ensure images are in `.jpg` or `.png` format and that labels are mapped in `train.csv`.
   
2. **Training**:
   - Run the notebook to train the model on the specified dataset.

3. **Evaluation**:
   - Use the evaluation code blocks to assess the model's accuracy and visualize performance with confusion matrices.
   
4. **Prediction and Export**:
   - After training, predictions for the test set are saved in `test_labels.csv` for potential integration into an inventory system.

## Conclusion
This CNN-based multilabel classification model performs well in identifying clothing type and color from images, making it an effective tool for e-commerce inventory management. The model's high accuracy and reliability position it for deployment in real-world scenarios, enhancing both operational efficiency and customer experience.
