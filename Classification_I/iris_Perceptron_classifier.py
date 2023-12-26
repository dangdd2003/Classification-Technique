import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()
iris_df = pd.DataFrame(
    data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
)
# print(iris_df.head())
# print(iris_df.describe())

# split X and Y
x = iris_df.iloc[:, :-1]
y = iris_df.iloc[:, -1]
# print(x.head())
# print(y.head())

# PCA
pca = PCA(n_components=2)
x = pca.fit_transform(x)

colors = iris_df["target"].replace(
    to_replace=[0, 1, 2], value=["red", "green", "green"]
)
plt.scatter(x[:, 0], x[:, 1], c=colors)

plt.show()

y.replace(to_replace=[0, 1, 2], value=[0, 1, 1], inplace=True)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, shuffle=True, random_state=0
)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

clf = Perceptron(max_iter=1000, eta0=0.1)  # eta0 is learning rate
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(y_pred)

print("confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("classifition report:")
print(classification_report(y_test, y_pred))
print("accuracy score:", accuracy_score(y_test, y_pred))
