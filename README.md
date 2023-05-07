# Convolutional Neural Network (CNN) for CIFAR-10 Image Classification

The project implements a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset.
## 1. Importing Libraries
The necessary libraries are imported at the beginning of the code, including TensorFlow, NumPy, Matplotlib, and scikit-learn's metrics module.

## 2. Loading and Preparing the Dataset
 
The CIFAR-10 dataset is loaded using the ***datasets.cifar10.load_data()*** function provided by TensorFlow. The dataset is split into training and test sets, containing images and corresponding labels. The shape of the training and test datasets is printed to verify the data dimensions. The class names for the labels are defined as a list.

## 3. Splitting Data into Training and Validation Sets
 
A fraction of the training data is split into a validation set using the ***validation_split parameter***. The code shuffles the indices of the training dataset and splits them into training and validation indices. The corresponding images and labels are extracted using the indices, creating the training and validation datasets.

## 4. Data Preprocessing
 
The pixel values of the images in both the training and test datasets are normalized by dividing them by 255. This step ensures that the pixel values are in the range of 0 to 1, which helps in better convergence during model training.

## 5. Data Augmentation
 
Applied data augmentation techniques, including random rotation, zooming, and horizontal flipping, to increase the diversity of the training data and improve the model's generalization.

## 6. CNN Model Architecture
 

The model is defined using the Sequential API provided by Keras (a part of TensorFlow). The model consists of the following layers:

***data_augmentation***: It is an instance of an image data augmentation layer. This layer applies random transformations to the input images during training, such as rotation, zooming, and flipping. Data augmentation helps in improving the model's ability to generalize by exposing it to a variety of variations of the input data.

***Conv2D layer (filters=64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)):*** This is the first convolutional layer with 64 filters (also known as feature detectors) of size 3x3. It uses the ReLU activation function to introduce non-linearity. The input shape of the images is (32, 32, 3), indicating images with a height and width of 32 pixels and three color channels (RGB).

***MaxPooling2D layer ((2, 2)):*** This layer performs max pooling operation with a pool size of 2x2, reducing the spatial dimensions of the feature maps by half. Max pooling helps in reducing the spatial size while retaining the important features.

***Second Conv2D layer (filters = 128, kernel_size= (3, 3), activation='relu'):*** This is the second convolutional layer with 128 filters of size 3x3. Again, it uses the ReLU activation function.

***Second MaxPooling2D layer ((2, 2)):*** Another max pooling layer is applied after the second convolutional layer.

***Flatten layer***: This layer flattens the 2D feature maps into a 1D vector, preparing the data for the fully connected layers.

***Dense layer (128, activation='relu'):*** This is a fully connected layer with 128 units and the ReLU activation function. It takes the flattened input and applies a linear transformation followed by the activation function.

***Dropout layer (0.5):*** Dropout is a regularization technique that randomly sets a fraction of input units to 0 during training. In this case, a dropout rate of 0.5 is applied, meaning 50% of the units will be randomly dropped during each training step. This helps in reducing overfitting and improving generalization.

***Final Dense layer (10, activation='softmax'):*** This is the output layer with 10 units, corresponding to the 10 classes in the CIFAR-10 dataset. It uses the softmax activation function to convert the raw scores into probabilities, indicating the likelihood of each class.

## 7. Compiling and Training the Model
 
 
The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss function (since the labels are integers), and accuracy as the metric to monitor during training. The model is then trained on the training dataset for a specified number of epochs (in this case, 10). The training history is stored in the history object.

## 8. Visualizing Training and Validation Curves
 
 
The code includes two plots to visualize the training and validation curves. The first plot shows the training accuracy over epochs, while the second plot shows the training loss over epochs. These plots help in understanding the model's performance during training.
 
 

## 9. Model Evaluation
 
The trained model is evaluated on the test dataset using the evaluate function. It calculates the test loss and test accuracy. The obtained accuracy is printed.

## 10. Making Predictions
 
 
The model is then used to make predictions on the test dataset using the predict function. The predictions are stored in the predictions variable.

## 11. Visualizing Predictions
 
A figure is created to visualize a grid of 25 images from the test dataset. For each image, the true class and predicted class (with the highest probability) are displayed as labels. This visualization helps in understanding how well the model performs on individual images.

## 12. Classification Report
The code generates a classification report using the ***classification_report*** function from scikit-learn's metrics module. It calculates various metrics like precision, recall, F1-score, and support for each class, as well as the overall accuracy. The classification report is printed to provide a comprehensive evaluation of the model's performance.
 

## 13. Confusion Matrix
A confusion matrix is plotted using the ***confusion_matrix*** function from scikit-learn's metrics module. The confusion matrix provides insights into the model.

