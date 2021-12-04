import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor


df = pd.read_csv("data/water_potability.csv")
col = list(df.columns)
col_target = col.pop()

df = df.fillna(0)

X_train, X_test, Y_train, Y_test = train_test_split(df[col],
                                    df[col_target], random_state=0)

clf = DecisionTreeClassifier(max_depth = 5, random_state = 0)
clf = clf.fit(X_train,Y_train)
tree.plot_tree(clf)
