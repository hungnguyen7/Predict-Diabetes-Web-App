#link : https://colab.research.google.com/drive/1f2-4xienoUVtfVNtyhiVOpj5P-TCN0MR#scrollTo=E_OUczBZUChA
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
model = tree.DecisionTreeClassifier(criterion="gini",max_depth=10)
model= model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Do chinh xac cua mo hinh voi nghi thuc kiem tra hold-out: %.3f" %
accuracy_score(y_test, y_pred))
# Hiển thị cây
tree.plot_tree(model.fit(X, y))
plt.show()
#Dùng graphviz để hiện thị to hơn các luật của cây.
import graphviz
dot_data=tree.export_graphviz(model, out_file=None,feature_names=None, class_names=y_pred, label='all',filled=False, leaves_parallel=False, impurity=True,node_ids=False, proportion=False, rotate=False,rounded=False, special_characters=False, precision=3)
graph =graphviz.Source(dot_data)
graph
#Nhìn ra được 2 luật điển hình 
#Luật 1:  X[0]<=34,x[2]=No,x[13]=No,X[14]=No ==> Negative với các thuộc tính khác là tùy tình trạng.
#Luật 2: Với độ tuổi X[1] là tùy lựa chọn, X[2]=Yes ,X[3]=Yes, X[7]=Yes ==> Positive với các thuộc tính khác là tùy tình trạng.