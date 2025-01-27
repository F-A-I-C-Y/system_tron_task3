# Decision Tree Classification

This project demonstrates the use of a **Decision Tree Classifier** to perform classification tasks. The workflow includes loading a dataset, training a decision tree model, and evaluating its performance.

## Dataset
The dataset used for this project must have relevant features and a target variable for classification. Please ensure the dataset is in CSV format and is available for use in the project.

## Objective
Train a Decision Tree Classifier to accurately predict the target variable based on the provided features.

## Files
- **Task 3 - Decision Tree.ipynb**: The Jupyter notebook containing the implementation of the decision tree model.
- **Dataset file (e.g., dataset.csv)**: The data used to train and evaluate the model.

## Requirements
The following Python libraries are required to run the project:
- pandas
- scikit-learn

## Steps to Run
1. Open the Jupyter notebook `Task 3 - Decision Tree.ipynb`.
2. Ensure the dataset file (e.g., `dataset.csv`) is in the same directory.
3. Follow the instructions in the notebook to execute each cell sequentially.
4. The notebook includes the following steps:
   - Data loading and preprocessing
   - Splitting the dataset into training and testing sets
   - Training the Decision Tree Classifier
   - Evaluating model performance (e.g., accuracy, precision, recall)

## Results
The notebook provides:
- A trained Decision Tree Classifier.
- Model evaluation metrics such as accuracy and a classification report.
- Visual representation of the decision tree (if included).

## Output
The trained model can be used for making predictions on new data. For example:
```python
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the saved model
model = joblib.load('decision_tree_model.pkl')

# Predict with new data
predictions = model.predict(new_data)
```

This project is a basic implementation of a Decision Tree Classifier and can be extended with:
- Hyperparameter tuning using GridSearchCV.
- Feature importance analysis.
- Exporting and visualizing the decision tree as a graph.

