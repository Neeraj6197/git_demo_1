import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# split the dataset into features and target variable
X = data.drop('target', axis=1)
y = data['target']

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data, data['target'], test_size=0.2, random_state=42
)

# Train a Random Forest Classifier
params = {'n_estimators': 300,
            'max_depth': 20}

model = RandomForestClassifier(**params)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")