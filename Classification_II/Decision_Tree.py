import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# import iris data set
iris = load_iris()
iris_df = pd.DataFrame(
    data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
)
print(iris["feature_names"])
print(iris["target_names"])
# print(iris_df.head())
# print(iris_df.describe())

# split X and Y (data and tartget)
x = iris_df.iloc[:, :-1]
y = iris_df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=100
)

# decision tree classification using entropy
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)

# make prediction
y_pred = dt.predict(x_test)

print(y_pred)

# calculate nessary metrics
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("classification report:\n", classification_report(y_test, y_pred))
print("accuracy score:", accuracy_score(y_test, y_pred) * 100)


# Plot the decision tree
def plot_decision_tree(clf_object, feature_names, class_name):
    plt.figure(figsize=(12, 8))
    plot_tree(
        clf_object,
        filled=True,
        rounded=True,
        feature_names=feature_names,
        class_names=class_name,
    )
    plt.show()


plot_decision_tree(dt, iris["feature_names"], iris["target_names"])
