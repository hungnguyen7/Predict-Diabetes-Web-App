# -*- coding: utf-8 -*-
"""DM_DoAn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FSF7kiTXyyHcZJl7HXnAh-EwswA0W_lc
"""




from sklearn import tree
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as  pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pickle
data=pd.read_csv("./diabetes_data_upload.csv")
#kiem tra null => Data không null!!
# print(data.isnan().sum())

data=data.replace('Male', 1)
data=data.replace('Female', 0)
data=data.replace('Yes', 1)
data=data.replace('No', 0)
target='class'
y=data[target]
X=data.drop(columns=['class'])

# Ti le giua Positive va Negative
print(y.value_counts(normalize=True))

#Tao model voi KNN
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5)
nFold=10
scores=cross_val_score(model, X, y, cv=nFold)
print("Do chinh xac cua mo hinh su dung K-Neighbors voi nghi thuc kiem tra %d-fold %.3f" %(nFold, np.mean(scores)))
#Tao model voi Decision Tree
from sklearn.tree import DecisionTreeClassifier
decisionTreeModel=[]
criterion=['gini', 'entropy']
for i in criterion:
  model2=DecisionTreeClassifier(criterion=i, max_depth=10, max_features='auto', random_state=1, splitter='best')
  model2.fit(X, y)
  decisionTreeModel.append(model2)
  scores = cross_val_score(model2, X, y, cv=nFold)
  print("Do chinh xac cua mo hinh su dung Decision Tree tham so %s voi nghi thuc kiem tra %d-fold %.3f" %(i, nFold, np.mean(scores)))

#Lay model decision Tree voi ket qua tot nhat
pickle.dump(decisionTreeModel[0], open('model.pkl', 'wb'))



#Luật Decision Tree
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
"""This stratify parameter makes a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to parameter stratify.

For example, if variable y is a binary categorical variable with values 0 and 1 and there are 25% of zeros and 75% of ones, stratify=y will make sure that your random split has 25% of 0's and 75% of 1's.
https://medium.com/dev-genius/early-stage-diabetes-risk-prediction-bd42f113a20b
"""