import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
    LeaveOneOut,
)
from sklearn.decomposition import PCA

# preparing dataset
iris = load_iris()
iris_df = pd.DataFrame(
    data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
)
print(iris_df.head())
# print(iris_df.describe())

# split X and Y
x = iris_df.iloc[:, :-1]
y = iris_df.iloc[:, -1]
print(x.head())
print(y.head())

# PCA
# pca = PCA(n_components=2)
# x = pca.fit_transform(x)

# split dataset into train and test set

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, shuffle=True, random_state=0
)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

print(
    f"training set size: {x_train.shape[0]} samples \ntest set size: {x_test.shape[0]} samples"
)

# normalize the dataset
# scaler = Normalizer().fit(x_train)  # the scaler is fitted to the train set
# x_train = scaler.transform(x_train)  # apply scaler to train set
# x_test = scaler.transform(x_test)  # apply scaler to test set
# print("x_train after normalization:")
# print(x_train)

# initialize kNN algorithm with library sklearn
x = 10
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
print(pred)
print(y_test)

# leave-one-out cross-validation
# cv = LeaveOneOut()
# scores = cross_val_score(knn, x, y, scoring="accuracy", cv=cv, n_jobs=-1)
# print(f"Accuracy: means: {np.mean(scores)}; std: {np.std(scores)}")

print(f"y_test = y_pred for k={x} ? ===> { np.array_equal(pred, y_test) }")

print("confusion matrix:")
print(confusion_matrix(y_test, pred))
print("classifition report:")
print(classification_report(y_test, pred))
print("accuracy score:", accuracy_score(y_test, pred))
