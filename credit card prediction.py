!pip install feature_engine

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import os
import time
import datetime
import sys
import sklearn
import scipy.stats as stats
import feature_engine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/content/drive/MyDrive/MACHINE LEARNING/ML_PROJECT/Credit card real time project/creditcard.csv')
df.sample(10)

**Preprocessing the data**
- `Find the data information Number of rows and columns `
- `Any Null values please check `
- `If there is any categorical column please find the labels and there count `
- `what is independent and what is dependent data `
- `Make Sure that we Split the data in the Initial Stage only `

from google.colab import drive
drive.mount('/content/drive')

df.info()

for i in df.columns:
  if df[i].isnull().sum() > 0:
    print(f'Number of Null values in Feature : {i} -> {df[i].isnull().sum()}')

df.tail(3)

df = df.drop([150000,150001],axis=0)

# checking null values
for i in df.columns:
  if df[i].isnull().sum() > 0:
    print(f'Number of Null values in Feature : {i} -> {df[i].isnull().sum()}')

for j in df.columns:
  if df[j].dtype == 'object':
    print(f'Number of Lables in {j} -> {df[j].unique()} -> {len(df[j].unique())} -> count : {df[j].value_counts()}')

x = df.iloc[: , :-1] # independent data
y = df.iloc[: , -1] # dependent

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=11)

print(f'number rows for training purpose : {len(x_train)}')
print(f'number rows for testing purpose : {len(x_test)}')

Handling Missing Values for Numerical Features

c = []
d = []
for i in x_train.columns:
  if x_train[i].isnull().sum() > 0:
    c.append(x_train[i].dtype)
    d.append(i)

print(c)
print(d)

x_train.head()

# checking whether MonthlyIncome and MonthlyIncome.1 same or not
f = []
for i in x_train.index:
  if x_train['MonthlyIncome'][i] == x_train['MonthlyIncome.1'][i]:
    pass
  elif np.isnan(x_train['MonthlyIncome'][i]) == np.isnan(x_train['MonthlyIncome.1'][i]):
    pass
  else:
    f.append(i)
print(f)

x_train = x_train.drop(['MonthlyIncome.1'],axis=1)

x_train.info()

c = []
d = []
for i in x_train.columns:
  if x_train[i].isnull().sum() > 0:
    c.append(x_train[i].dtype)
    d.append(i)
print(c)
print(d)

- `Mean Median Mode **`
- `End of Distribution - `
- `Arbitary Value -`
- `Constant value technique `
- `Random Sample Technique *********`

x_train.isnull().sum()

mean_MonthlyIncome = x_train['MonthlyIncome'].mean()
mean_MonthlyIncome

median_MonthlyIncome = x_train['MonthlyIncome'].median()
median_MonthlyIncome

mode_MonthlyIncome = x_train['MonthlyIncome'].mode()[0]
mode_MonthlyIncome

def fun(col,x,y,z):
  x_train[col+'_mean'] = x_train[col].fillna(x)
  x_train[col+'_median'] = x_train[col].fillna(y)
  x_train[col+'_mode'] = x_train[col].fillna(z)

fun('MonthlyIncome',mean_MonthlyIncome,median_MonthlyIncome,mode_MonthlyIncome)

x_train.isnull().sum()

print(f'the std of original monthly income : {x_train["MonthlyIncome"].std()}')
print(f'the std of mean monthly income : {x_train["MonthlyIncome_mean"].std()}')
print(f'the std of median monthly income : {x_train["MonthlyIncome_median"].std()}')
print(f'the std of mode monthly income : {x_train["MonthlyIncome_mode"].std()}')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

x_train['MonthlyIncome'].plot(kind = 'kde' , ax = ax , label = 'original')
x_train['MonthlyIncome_mean'].plot(kind = 'kde' , ax = ax , label = 'mean')
x_train['MonthlyIncome_median'].plot(kind = 'kde' , ax = ax , label = 'median')
x_train['MonthlyIncome_mode'].plot(kind = 'kde' , ax = ax , label = 'mode')

plt.legend(loc=0)
plt.show()

x_train = x_train.drop(['MonthlyIncome','MonthlyIncome_mean','MonthlyIncome_median'],axis=1)

x_train.isnull().sum()

checking with X_test

x_test.isnull().sum()

x_test['MonthlyIncome'] = x_test['MonthlyIncome'].fillna(mode_MonthlyIncome)
x_test.isnull().sum()

x_test = x_test.drop(['MonthlyIncome.1'],axis=1)

x_test.isnull().sum()

# again coming back to the training data
x_train['NumberOfDependents']

# converting string data to int
x_train['NumberOfDependents'] = pd.to_numeric(x_train['NumberOfDependents'])
x_train['NumberOfDependents'].dtype

number_mean,number_median,number_mode = x_train['NumberOfDependents'].mean(),x_train['NumberOfDependents'].median(),x_train['NumberOfDependents'].mode()[0]

fun('NumberOfDependents',number_mean,number_median,number_mode)

x_train.isnull().sum()

print(f'the std od original NumberOfDependencies : {x_train["NumberOfDependents"].std()}')
print(f'the std of NumberOfDependents_mean : {x_train["NumberOfDependents_mean"].std()}')
print(f'the std of NumberOfDependents_median : {x_train["NumberOfDependents_median"].std()}')
print(f'the std of NumberOfDependents_mode : {x_train["NumberOfDependents_mode"].std()}')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

x_train['NumberOfDependents'].plot(kind = 'kde',ax = ax,label = 'original')
x_train['NumberOfDependents_mode'].plot(kind = 'kde' ,ax = ax , label = 'mode')
x_train['NumberOfDependents_mean'].plot(kind = 'kde' ,ax = ax , label = 'mean')
x_train['NumberOfDependents_median'].plot(kind = 'kde' ,ax = ax , label = 'median')

plt.legend(loc = 0)
plt.show()

fig = plt.figure()

ax = fig.add_subplot(1,1,1)


x_train['NumberOfDependents'].hist(bins=10,ax=ax,label='original')
x_train['NumberOfDependents_mode'].hist(bins=10,ax=ax,label = 'mode')
x_train['NumberOfDependents_mean'].hist(bins=10,ax=ax,label = 'mean')
x_train['NumberOfDependents_median'].hist(bins=10,ax=ax,label = 'Median')

plt.legend(loc = 0)

plt.show()

x_train.isnull().sum()

x_train = x_train.drop(['NumberOfDependents','NumberOfDependents_mean','NumberOfDependents_mode'],axis=1)

x_train.isnull().sum()

x_test['NumberOfDependents'] = pd.to_numeric(x_test['NumberOfDependents'])
x_test['NumberOfDependents'] = x_test['NumberOfDependents'].fillna(number_median)

x_test.isnull().sum()

`so we have completed -> Null values cleaning in Numerical columns and there are no Null values in categorical columns `

x_train.sample(5)

# I am selecting Only Numerical Columns and going to check Normal Distribution | Variable Transformation | Feature Scaling | Outliers  Handlling

numerical_x_train = x_train.select_dtypes(exclude='object')
numerical_x_train

import scipy.stats as stats
def n_d(numerical_x_train,var):
  plt.figure(figsize=(8,3))
  plt.subplot(1,3,1)
  plt.title(str(var))
  numerical_x_train[var].plot(kind='kde',color = 'g')
  plt.subplot(1,3,2)
  plt.title(str(var))
  stats.probplot(numerical_x_train[var],dist='norm',plot=plt)
  plt.subplot(1,3,3)
  sns.boxplot(x = numerical_x_train[var])
  plt.show()

for i in numerical_x_train.columns:
  n_d(numerical_x_train,i)

**Variable Tranformation**

- `Log Transorformation `
- `1/n Transormation `
`- square root Transformation `
- `Exponentional Transformation `
- `Box Cox Transormation `
- `Yeojohnson Transformation `

# so we are going to apply log and convert the data into best way
def log(numerical_x_train,original,log_original):
  plt.figure(figsize=(8,3))
  plt.subplot(1,2,1)
  plt.title('original_numerical_x_train_feature')
  numerical_x_train[original].plot(kind='kde',color='r',label='original_x_train_feature')
  plt.subplot(1,2,2)
  plt.title('log_feature_trasfermation')
  numerical_x_train[log_original].plot(kind='kde',color='g',label='log_feature')
  plt.show()



for i in numerical_x_train.columns:
  numerical_x_train[i+'_log'] = np.log(numerical_x_train[i] + 1)
  log(numerical_x_train,i,i+'_log')

numerical_x_train.columns

# checking outliers for originalX_train_num features and converted [log] outliers
import warnings
warnings.filterwarnings('ignore')

def boxplot_(numerical_x_train,original,log_original):
  plt.figure(figsize = (8,3))
  plt.subplot(1,2,1)
  plt.title('original_x_train_numerical_feature')
  sns.boxplot(x = numerical_x_train[original])
  plt.subplot(1,2,2)
  plt.title('Log_feature_Transformation')
  sns.boxplot(x = numerical_x_train[log_original])
  plt.show()

c = []
d = []
for i in numerical_x_train.columns:
  if i.endswith('_log'):
    c.append(i)
  else:
    d.append(i)
for j in range(len(c)):
    boxplot_(numerical_x_train,d[j],c[j])

numerical_x_train = numerical_x_train.drop(['NPA Status', 'RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'MonthlyIncome_mode', 'NumberOfDependents_median'],axis=1)

numerical_x_train.columns

numerical_x_train.sample(6)

numerical_x_train = numerical_x_train.drop(['NPA Status_log','NumberOfTime30-59DaysPastDueNotWorse_log'],axis=1)

numerical_x_train.columns

to handle outliers I am using 5th and 95th quantile

# 5th and 95th
def fifth(numerical_x_train,var):
  upper = numerical_x_train[var].quantile(0.95)
  lower = numerical_x_train[var].quantile(0.05)
  return upper,lower

for i in numerical_x_train.columns:
  upper,lower = fifth(numerical_x_train,i)
  numerical_x_train[i+'_5th'] = np.where(numerical_x_train[i] > upper, upper,
                                  np.where(numerical_x_train[i] < lower, lower, numerical_x_train[i]))

def boxplot_(numerical_x_train,original,log_original):
  plt.figure(figsize = (8,3))
  plt.subplot(1,2,1)
  plt.title('original_x_train_numerical_feature')
  sns.boxplot(x = numerical_x_train[original])
  plt.subplot(1,2,2)
  plt.title('5th_feature_Transformation')
  sns.boxplot(x = numerical_x_train[log_original])
  plt.show()

c = []
d = []
for i in numerical_x_train.columns:
  if i.endswith('_log'):
    c.append(i)
  else:
    d.append(i)

for j in range(len(c)):
  boxplot_(numerical_x_train,c[j],d[j])

numerical_x_train.columns

numerical_x_train = numerical_x_train.drop(['RevolvingUtilizationOfUnsecuredLines_log', 'age_log', 'DebtRatio_log',
       'NumberOfOpenCreditLinesAndLoans_log', 'NumberOfTimes90DaysLate_log',
       'NumberRealEstateLoansOrLines_log',
       'NumberOfTime60-89DaysPastDueNotWorse_log', 'MonthlyIncome_mode_log',
       'NumberOfDependents_median_log'],axis=1)

numerical_x_train.columns

a1 = ['DebtRatio_log_5th','NumberOfTimes90DaysLate_log_5th','NumberOfTime60-89DaysPastDueNotWorse_log_5th']
for j in a1:
  print(numerical_x_train[j].unique())

numerical_x_train = numerical_x_train.drop(['NumberOfTimes90DaysLate_log_5th','NumberOfTime60-89DaysPastDueNotWorse_log_5th'],axis=1)

numerical_x_train.columns

# same things which we applied on the numerical_X_train need to implement on the X_test
x_test.columns

x_test = x_test.drop(['NPA Status','NumberOfTime30-59DaysPastDueNotWorse','NumberOfTimes90DaysLate','NumberOfTime60-89DaysPastDueNotWorse'],axis=1)
x_test.columns

numerical_x_test = x_test.select_dtypes(exclude='object')
numerical_x_test

# so we are going to apply log and convert the data into best way
def log(numerical_x_test,original,log_feature):
  plt.figure(figsize=(8,3))
  plt.subplot(1,2,1)
  plt.title('original_numerical_x_test_feature')
  numerical_x_test[original].plot(kind = 'kde',color = 'g', label = 'original_feature')
  plt.legend(loc=0)
  plt.subplot(1,2,2)
  plt.title('numerical_log_feature')
  numerical_x_test[log_feature].plot(kind = 'kde',color = 'b',label = 'log_feature')
  plt.legend(loc=0)
  plt.show()

for i in numerical_x_test.columns:
  numerical_x_test[i+'_log'] = np.log(numerical_x_test[i] + 1)
  log(numerical_x_test,i,i+'_log')

numerical_x_test.columns

numerical_x_test = numerical_x_test.drop(['RevolvingUtilizationOfUnsecuredLines', 'age', 'MonthlyIncome','DebtRatio', 'NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines', 'NumberOfDependents'],axis=1)

numerical_x_test.columns

# 5th and 95th
def fifth(numerical_x_test,var):
  upper = numerical_x_test[var].quantile(0.95)
  lower = numerical_x_test[var].quantile(0.05)
  return upper,lower

for i in numerical_x_test.columns:
  upper,lower = fifth(numerical_x_test,i)
  numerical_x_test[i+'_5th'] = np.where(numerical_x_test[i] > upper,upper,
                                       np.where(numerical_x_test[i] < lower,lower , numerical_x_test[i]))


numerical_x_test.columns

# checking outliers for originalX_train_num features and converted [log] outliers
def boxplot(numerical_x_test,original,fifth_and_95th):
  plt.figure(figsize=(8,3))
  plt.subplot(1,2,1)
  plt.title('log_feature_numerical_x_test')
  sns.boxplot(x = numerical_x_test[original])
  plt.legend(loc=0)
  plt.subplot(1,2,2)
  plt.title('5th_feature_numerical_x_test')
  sns.boxplot(x = numerical_x_test[fifth_and_95th])
  plt.legend(loc=0)
  plt.show()

c = []
d = []
for i in numerical_x_test.columns:
  if i.endswith('_log'):
    c.append(i)
  else:
    d.append(i)
for j in range(len(c)):
   boxplot_(numerical_x_test,c[j],d[j])

numerical_x_test = numerical_x_test.drop(['RevolvingUtilizationOfUnsecuredLines_log', 'age_log',
       'MonthlyIncome_log', 'DebtRatio_log',
       'NumberOfOpenCreditLinesAndLoans_log',
       'NumberRealEstateLoansOrLines_log', 'NumberOfDependents_log'],axis=1)
numerical_x_test.columns

len(numerical_x_train.columns),len(numerical_x_test.columns)

- `Since in the Training data and Test Data we have cleared null values | varibale transformation and maintained Normal Distribution | and also handled Outliers ... -> finally in both X_train[numeircal columns] and X_test[numerical columns] issues are solved`

`Now we are going to work with X_train[categorical Data to Numerical data] -> Even in the X_test -> [categorical data to Numerical data ]`

categorical_x_train = x_train.select_dtypes(include = 'object')
categorical_x_train.head(5)

# since Gender and Region Features are Nominal Encoding -> we will use OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
one_hot = OneHotEncoder(handle_unknown='ignore')
one_hot.fit(categorical_x_train[['Gender']])
b = one_hot.transform(categorical_x_train[['Gender']]).toarray()
one_hot.categories_
categorical_x_train['Gender_male'] = b[: , 1].astype(int)

categorical_x_train.head()

one_hot_r = OneHotEncoder(handle_unknown='ignore')
one_hot_r.fit(categorical_x_train[['Region']])
b = one_hot_r.transform(categorical_x_train[['Region']]).toarray()
one_hot_r.categories_[0][0]
categorical_x_train[one_hot_r.categories_[0][0]] =  b[: , 0].astype(int)
categorical_x_train[one_hot_r.categories_[0][1]] =  b[: , 1].astype(int)
categorical_x_train[one_hot_r.categories_[0][2]] =  b[: , 2].astype(int)
categorical_x_train[one_hot_r.categories_[0][3]] =  b[: , 3].astype(int)

categorical_x_train.head()

# apply odinal encoding to rented house occupation and education
from sklearn.preprocessing import OrdinalEncoder
od_r = OrdinalEncoder()
od_r.fit(categorical_x_train[['Rented_OwnHouse']])
categorical_x_train['Rented'] = od_r.transform(categorical_x_train[['Rented_OwnHouse']]).astype(int)
categorical_x_train.head()

# apply odinal encoding to rented house occupation and education
from sklearn.preprocessing import OrdinalEncoder
od_o = OrdinalEncoder()
od_o.fit(categorical_x_train[['Occupation']])
categorical_x_train['Occupation_re'] = od_o.transform(categorical_x_train[['Occupation']]).astype(int)
categorical_x_train.head()

# apply odinal encoding to rented house occupation and education
from sklearn.preprocessing import OrdinalEncoder
od_e = OrdinalEncoder()
od_e.fit(categorical_x_train[['Education']])
categorical_x_train['Education_re'] = od_e.transform(categorical_x_train[['Education']]).astype(int)
categorical_x_train.head()

categorical_x_train = categorical_x_train.drop(['Gender','Region','Rented_OwnHouse','Occupation','Education'],axis=1)
categorical_x_train.head()

Same Techniques -> we are going to implement in X_test categorical data ...

numerical_x_test.head()

categorical_x_test = x_test.select_dtypes(include = 'object')
categorical_x_test.head()

# already we have implemented in training data -> same we are going to implement in test data

b = one_hot.transform(categorical_x_test[['Gender']]).toarray()
categorical_x_test['gender_male'] = b[: , 1].astype(int)


b1 = one_hot_r.transform(categorical_x_test[['Region']]).toarray()

categorical_x_test[one_hot_r.categories_[0][0]] = b1[: , 0].astype(int)
categorical_x_test[one_hot_r.categories_[0][1]] = b1[: , 1].astype(int)
categorical_x_test[one_hot_r.categories_[0][2]] = b1[: , 2].astype(int)
categorical_x_test[one_hot_r.categories_[0][3]] = b1[: , 3].astype(int)


categorical_x_test['Rented'] = od_r.transform(categorical_x_test[['Rented_OwnHouse']]).astype(int)

categorical_x_test['Occupation_re'] = od_o.transform(categorical_x_test[['Occupation']]).astype(int)

categorical_x_test['Education_re'] = od_e.transform(categorical_x_test[['Education']]).astype(int)

categorical_x_test

categorical_x_test = categorical_x_test.drop(['Gender','Region','Rented_OwnHouse','Occupation','Education'],axis=1)

categorical_x_test

making all training data into 1 part and test data into 1 part

x_train_perfect = pd.concat([numerical_x_train,categorical_x_train],axis=1)
x_train_perfect.sample(5)

x_test_perfect = pd.concat([numerical_x_test,categorical_x_test],axis=1)
x_test_perfect.sample(5)

print(x_test_perfect.shape)

Feature_selection

# constant technique and quasi constant

from sklearn.feature_selection import VarianceThreshold
reg = VarianceThreshold(threshold=0) # defaulty variance 0


reg.fit(x_train_perfect)

sum(reg.get_support())   # 267 feturess are not constant

constant = x_train_perfect.columns[~reg.get_support()]

len(constant)

constant

x_train_perfect.shape

x_train_perfect.head()

y_train.head()

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
lb.fit(y_train)

y_train_n = lb.transform(y_train)
y_train_n[:10]

y_test_n = lb.transform(y_test)
y_test_n[:10]

x_train_perfect

feature scaling

# Before going to train the algo we need to scale down the values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train_perfect)
x_train_perfect_s = sc.transform(x_train_perfect)
x_train_perfect_s

x_train_perfect.columns

x_test_perfect.columns

x_test_perfect_s = x_test_perfect[['RevolvingUtilizationOfUnsecuredLines_log_5th', 'age_log_5th','DebtRatio_log_5th', 'NumberOfOpenCreditLinesAndLoans_log_5th','NumberRealEstateLoansOrLines_log_5th', 'MonthlyIncome_log_5th','NumberOfDependents_log_5th', 'gender_male', 'Central', 'East','North', 'South', 'Rented', 'Occupation_re', 'Education_re']]

x_test_perfect_s.columns

x_test_perfect_s = x_test_perfect_s.rename(columns={'MonthlyIncome_log_5th':'MonthlyIncome_mode_log_5th' , 'NumberOfDependents_log_5th':'NumberOfDependents_median_log_5th'})

x_test_perfect_s.columns

# since the train data is scale down we can give the data to the algorithm
# KNN , Naive bayes , Logistic Regression , Decision Tree , Random Forest

def KNN(x_train,y_train,x_test,y_test):
  reg_knn = KNeighborsClassifier(n_neighbors=5)
  reg_knn.fit(x_train,y_train)
  print(f'training_accuracy : {reg_knn.score(x_train,y_train)}')
  print(f'testing_accuracy : {reg_knn.score(x_test,y_test)}')
  print(f'confusion_matrix : {confusion_matrix(y_test,reg_knn.predict(x_test))}')
  print(f'confusion_matrix : {classification_report(y_test,reg_knn.predict(x_test))}')

def NB(x_train,y_train,x_test,y_test):
  reg_nb = GaussianNB()
  reg_nb.fit(x_train,y_train)
  print(f'training_accuracy : {reg_nb.score(x_train,y_train)}')
  print(f'testing_accuracy : {reg_nb.score(x_test,y_test)}')
  print(f'confusion_matrix : {confusion_matrix(y_test,reg_nb.predict(x_test))}')
  print(f'confusion_matrix : {classification_report(y_test,reg_nb.predict(x_test))}')

def LR(x_train,y_train,x_test,y_test):
  reg_lr = LogisticRegression()
  reg_lr.fit(x_train,y_train)
  print(f'training_accuracy : {reg_lr.score(x_train,y_train)}')
  print(f'testing_accuracy : {reg_lr.score(x_test,y_test)}')
  print(f'confusion_matrix : {confusion_matrix(y_test,reg_lr.predict(x_test))}')
  print(f'confusion_matrix : {classification_report(y_test,reg_lr.predict(x_test))}')

def DT(x_train,y_train,x_test,y_test):
  reg_dt = DecisionTreeClassifier(criterion='entropy')
  reg_dt.fit(x_train,y_train)
  print(f'training_accuracy : {reg_dt.score(x_train,y_train)}')
  print(f'testing_accuracy : {reg_dt.score(x_test,y_test)}')
  print(f'confusion_matrix : {confusion_matrix(y_test,reg_dt.predict(x_test))}')
  print(f'confusion_matrix : {classification_report(y_test,reg_dt.predict(x_test))}')

def RF(x_train,y_train,x_test,y_test):
  reg_rf = RandomForestClassifier(n_estimators=5)
  reg_rf.fit(x_train,y_train)
  print(f'training_accuracy : {reg_rf.score(x_train,y_train)}')
  print(f'testing_accuracy : {reg_rf.score(x_test,y_test)}')
  print(f'confusion_matrix : {confusion_matrix(y_test,reg_rf.predict(x_test))}')
  print(f'confusion_matrix : {classification_report(y_test,reg_rf.predict(x_test))}')

def calling(x_train,y_train,x_test,y_test):
    print('----knn---')
    KNN(x_train,y_train,x_test,y_test)
    print('---Naive bayes------')
    NB(x_train,y_train,x_test,y_test)
    print('---Logistic Regresssion----')
    LR(x_train,y_train,x_test,y_test)
    print('-----Decision Tree-------')
    DT(x_train,y_train,x_test,y_test)
    print('----Random Forest--------')
    RF(x_train,y_train,x_test,y_test)

calling(x_train_perfect_s,y_train_n,x_test_perfect_s,y_test_n)

# insted of accuracy we will check AUC and ROC and will decide the Model
reg_knn = KNeighborsClassifier()
reg_NB = GaussianNB()
reg_LR = LogisticRegression()
reg_DT = DecisionTreeClassifier(criterion='entropy')
reg_RF = RandomForestClassifier(n_estimators=5)


reg_knn.fit(x_train_perfect_s,y_train_n)
reg_NB.fit(x_train_perfect_s,y_train_n)
reg_LR.fit(x_train_perfect_s,y_train_n)
reg_DT.fit(x_train_perfect_s,y_train_n)
reg_RF.fit(x_train_perfect_s,y_train_n)


y_pred_knn = reg_knn.predict(x_test_perfect_s)
y_pred_nb = reg_NB.predict(x_test_perfect_s)
y_pred_lr = reg_LR.predict(x_test_perfect_s)
y_pred_dt = reg_DT.predict(x_test_perfect_s)
y_pred_rf = reg_RF.predict(x_test_perfect_s)

from sklearn.metrics import roc_curve,auc,roc_auc_score
fprk,tprk,threk = roc_curve(y_test_n,y_pred_knn)
fprn,tprn,thren = roc_curve(y_test_n,y_pred_nb)
fprl,tprl,threl = roc_curve(y_test_n,y_pred_lr)
fprd,tprd,thred = roc_curve(y_test_n,y_pred_dt)
fprr,tprr,threr = roc_curve(y_test_n,y_pred_rf)

plt.figure(figsize=(5,3))
plt.plot([0, 1], [0, 1], "k--")

plt.plot(fprk,tprk,color='r',label='knn')
plt.plot(fprn,tprn,color='black',label='NB')
plt.plot(fprl,tprl,color='g',label='LR')
plt.plot(fprd,tprd,color='y',label='dt')
plt.plot(fprr,tprr,color='blue',label='rf')

plt.legend(loc=0)
plt.show()

# since Finalizeed Model was Naive Bayes

# save the Model

import pickle

with open('credit_final_model.pkl','wb') as f:
    pickle.dump(reg_NB,f)
