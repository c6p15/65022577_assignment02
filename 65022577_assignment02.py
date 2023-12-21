from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV,\
RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.svm import SVC
import numpy as np

File_Path = 'D:/65022577/data/'
File_Name = 'Iris.xlsx'

df = pd.read_excel(File_Path + File_Name)

df.drop(columns=['Id'], inplace=True)
 
x = df.iloc[:, 0:4]
y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(x,y)

param_Grid = {
    'n_estimators': [25,50,100,150],
    'max_features': ['sqrt','log2',None],
    'criterion':['gini','entropy'],
    'max_depth':[3,6,9],
    'max_leaf_nodes':[3,6,9]
    }

#Create model
forest = GridSearchCV(RandomForestClassifier(), param_grid= param_Grid)
forest.fit(x_train,y_train)

#Vertify model
score_test = forest.score(x_test,y_test)
print('Accuracy :', '{:.2f}'.format(score_test))
 
Best_Parameter = forest.best_params_

model = DecisionTreeClassifier(criterion='entropy')
model.fit(x,y)

feature = x.columns.tolist()
Data_class = y.tolist()

plt.figure(figsize=(25,20))
_ = plot_tree(model,
              feature_names= feature,
              class_names = Data_class,
              label='all',
              impurity=True,
              precision=3,
              filled=True,
              rounded=True,
              fontsize=16)

plt.show()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
Label = y_train.unique()
Confu = confusion_matrix(y_test, y_pred)
CM_view = ConfusionMatrixDisplay(confusion_matrix = Confu,
                                 display_labels= Label)
CM_view.plot()
plt.show()


import seaborn as sns
Feature_imp = model.feature_importances_
feature_names = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'] 
 
sns.set(rc = {'figure.figsize' : (11.7,8.7)})
sns.barplot(x = Feature_imp, y = feature_names)