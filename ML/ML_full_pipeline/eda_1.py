# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:21:54 2023

@author: sanat
"""
# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

# %%
data = pd.read_csv("data/adult.csv")
data.head()
# %%
data.columns
# %%
print("The null values count :")
print((data == "?").sum())
print(f"the total no of samples are -> {len(data)}")
# %%
undefined_workclass = data[data['workclass'] == '?']
for a in undefined_workclass[["race", "age"]].groupby("race"):
    print(a)
undefined_workclass
# %%
df1 = data['workclass'].value_counts()
labels = df1.index
numbers = df1.values
df1[-1], df1[3] = df1[3], df1[-1]

fig, ax = plt.subplots()
plt.figure(figsize=(7, 5))
ax.pie(numbers, labels=labels, labeldistance=1.1, radius=2, autopct='%1.2f%%')
# %%
data.describe()
# %%
