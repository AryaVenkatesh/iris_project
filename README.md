# iris_project
Iris Flower Classification using Logistic Regression

Project Overview: Iris Flower Classification using Logistic Regression
🔹 Introduction
This project aims to build a machine learning model that can accurately classify iris flowers into three species:
Setosa


Versicolor


Virginica


The classification is performed using the famous Iris dataset, which contains measurements of different flower parts such as sepal length, sepal width, petal length, and petal width.
 The main goal is to train a Logistic Regression model and evaluate its performance using various metrics.

🔹 Objective of the Project
The objective of this project is to:
Load and explore the Iris dataset


Perform Exploratory Data Analysis (EDA)


Preprocess the data


Build a Logistic Regression model


Evaluate the model using metrics such as accuracy, confusion matrix, and classification report



🔹 Description of the Steps
1. Importing Required Libraries
The project uses essential Python libraries such as:
NumPy and Pandas → for data handling


Seaborn and Matplotlib → for data visualization


Scikit-learn → for data preprocessing, model training, and evaluation


These libraries together make it easy to manage and analyze the dataset.

2. Loading the Iris Dataset
The dataset is loaded using:
iris = load_iris()

X contains the features (measurements of petals and sepals).


y contains the target labels (three species).


Labels are mapped to their species names for better interpretation.



3. Exploratory Data Analysis (EDA)
a) Pairplot
A pairplot is used to visualize the relationship between each pair of features.
 It helps understand:
Patterns


Separability between species


Feature distributions


b) Correlation Heatmap
This shows how strongly each feature is correlated with the others.
 For example:
Petal length and petal width are highly correlated


Sepal measurements have weaker correlations


EDA helps determine which features are most useful for classification.

4. Splitting and Preprocessing the Data
The dataset is split into:
80% Training data


20% Testing data


Then, features are standardized using StandardScaler:
Ensures all features have similar scales


Helps Logistic Regression converge faster and perform better



5. Building the Logistic Regression Model
A Logistic Regression model is trained using the scaled training data:
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

This model learns patterns and relationships within the features to classify the species.

6. Model Evaluation
Predictions are made using the test dataset.
 The following evaluation metrics are used:
a) Confusion Matrix
Shows how many predictions were correctly or incorrectly classified for each species.
b) Classification Report
Provides:
Precision


Recall


F1-score


This helps judge the performance for each individual class.
c) Accuracy Score
Shows the overall percentage of correctly predicted labels.
Logistic Regression typically performs very well on this dataset due to its simplicity and linear separability.

🔹 Conclusion
This project successfully demonstrates the complete workflow of a machine learning classification task:
Data loading and cleanup


Visual exploration


Feature scaling


Model training


Performance evaluation


The Logistic Regression model achieves high accuracy on the Iris dataset, making it an effective choice for this classification problem.



