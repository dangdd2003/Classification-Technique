import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree

wine = load_wine()
wine_df = pd.DataFrame(
    data=np.c_[wine["data"], wine["target"]], columns=wine["feature_names"] + ["target"]
)
# print(wine["feature_names"])
# print(wine["target_names"])
# print(wine_df.describe())

# split X and Y (data and tartget)
x = wine_df.iloc[:, :-1]
y = wine_df.iloc[:, -1]
# print(x)
# print(y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
# Apply the Random Forest Classifier
random_forests = RandomForestClassifier(n_estimators=100, criterion="entropy")
random_forests.fit(x_train, y_train)
# print(random_forests.predict(x_test))

# Apply Decision Tree Classifier
decision_trees = DecisionTreeClassifier(criterion="entropy")
decision_trees.fit(x_train, y_train)
# print(decision_trees.predict(x_test))

# Random Forest Classifier but using bagging and then decision tree
bagging = BaggingClassifier(estimator=decision_trees, n_estimators=100)
bagging.fit(x_train, y_train)
# print(bagging.predict(x_test))


plt.figure(figsize=(12, 8))
plot_tree(
    bagging.estimators_[100],
    # decision_trees,
    # random_forests.estimators_[0],
    filled=True,
    rounded=True,
    feature_names=wine["feature_names"],
    class_names=wine["target_names"],
)
plt.show()

# calculate nessary metrics
print("Accuracy Score:", accuracy_score(y_test, bagging.predict(x_test)) * 100)
print("Confusion Matrix:\n", confusion_matrix(y_test, bagging.predict(x_test)))
print(
    "Classification Report:\n", classification_report(y_test, bagging.predict(x_test))
)
