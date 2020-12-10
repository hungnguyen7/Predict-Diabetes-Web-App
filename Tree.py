import pandas as  pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pickle

from sklearn import tree
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("./diabetes_data_upload.csv")
train_validation, test=train_test_split(data, test_size=0.2, stratify=data['class'], random_state=17)
train, validation=train_test_split(train_validation, test_size=0.2, stratify=train_validation['class'], random_state=17)
data=data.replace('Male', 1)
data=data.replace('Female', 0)
data=data.replace('Yes', 1)
data=data.replace('No', 0)
target='class'
y=data[target]
X=data.drop(columns=['class'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = tree.DecisionTreeClassifier(criterion="gini")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Do chinh xac cua mo hinh voi nghi thuc kiem tra hold-out: %.3f" %
accuracy_score(y_test, y_pred))
# Hiển thị cây
tree.plot_tree(model.fit(X, y))
plt.show()
=======
plt.show()
import graphviz
dot_data=tree.export_graphviz(model,feature_names=None,class_names=None,filled=True,rounded=True,special_characters=True)

graph =graphviz.Source(dot_data)
graph
