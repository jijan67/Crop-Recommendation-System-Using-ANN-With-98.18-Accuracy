# 🌾Crop Recommendation System Using ANN (Artificial Neural Network)
This documentation provides a detailed explanation of the Crop Recommendation System project implemented using a neural network in Python. The goal of this system is to recommend the most suitable crop to grow based on environmental and soil conditions using machine learning techniques.
---
## Overview
The project employs a dataset containing features like Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH, and rainfall, with a target label representing different crop types. The model uses an Artificial Neural Network (ANN) for classification and achieves an accuracy of 98.18%.

## Libraries Used
The following Python libraries were used to implement the model:
- **NumPy:** For numerical operations.
- **Pandas:** For data manipulation and analysis.
- **Matplotlib & Seaborn:** For data visualization.
- **TensorFlow/Keras:** For building and training the neural network model.
- **Scikit-learn:** For data preprocessing and evaluation metrics.

## Project Structure
**1. Data Loading and Preprocessing:**
- The dataset is loaded from a CSV file, and the target variable (**label**) is encoded using **LabelEncoder**.
- Features are standardized using **StandardScaler**.
- The dataset is split into training and test sets using **train_test_split**.

**2. Model Architecture:**

- A **Sequential** model is created with:
  - Three fully connected (Dense) layers.
  - **ReLU** activation function for hidden layers and **Softmax** for the output layer (since this is a multi-class classification problem).
- The model is compiled with the **Adam optimizer, Sparse Categorical Crossentropy** loss function, and accuracy as the evaluation metric.

**3. Model Training:**

- The model is trained for **100 epochs** with a batch size of 32, and both training and validation data are provided during the training process.

- The training history (accuracy and loss) is recorded for visualization.

**4. Model Evaluation:**

- The model's accuracy and loss are plotted for both training and validation sets.

- A classification report, confusion matrix, and ROC (Receiver Operating Characteristic) curve are generated to evaluate the model's performance.

**5. Crop Recommendation Function:**

- A function recommend_crop is implemented to take user input (soil and environmental parameters) and recommend a crop based on the trained model.
## Evaluation Results
- **Test Accuracy:** 98.18%
- The model demonstrates strong performance with both training and validation accuracy consistently above 97% after approximately 20 epochs. The accuracy curve shows that the model quickly converges, with minimal signs of overfitting, as the training and validation accuracies remain close throughout the training.
- The confusion matrix shows that the model performs well, with minimal misclassifications.
- The ROC curve confirms that the model distinguishes between classes effectively, with high AUC values for each class.

## Conclusion
The crop recommendation system built with an artificial neural network demonstrates high accuracy (98.18%) in recommending the most suitable crop based on various environmental and soil conditions. This model can help farmers make informed decisions regarding crop selection based on real-time data.
---
