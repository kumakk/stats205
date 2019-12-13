#!/usr/bin/env python
# coding: utf-8

# In[303]:


import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report


# In[60]:


pets = pd.read_csv("train.csv")
pets_X = pets.iloc[:, :-1]
pets_y = pets.iloc[:, -1]


# In[71]:


def generatesentiment(text):
    sid = SentimentIntensityAnalyzer()
    score = sid.polarity_scores(text)
    return score


# In[72]:


pets_X['Description_Sentiment'] = ''


# In[73]:


for i in range(0, len(pets_X)):
    if i % 1000 == 0:
        print(i)
    pets_X['Description_Sentiment'][i] = generatesentiment(str(pets['Description'][i]))['compound']


# In[297]:


plt.figure(figsize=(10,8))
sns.countplot(x='AdoptionSpeed', data=pets)
plt.title("Number of examples per class")


# In[126]:


X_train, X_test, y_train, y_test = train_test_split(pets_X, pets_y, test_size=0.1, random_state=205)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=205)


# In[80]:


# KNN
X_train_knn = X_train[['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'Description_Sentiment']]
X_train_knn = sklearn.preprocessing.scale(X_train_knn)
X_val_knn = X_val[['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'Description_Sentiment']]
X_val_knn = sklearn.preprocessing.scale(X_val_knn)
X_test_knn = X_test[['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'Description_Sentiment']]
X_test_knn = sklearn.preprocessing.scale(X_test_knn)


# In[306]:


neigh = KNeighborsClassifier(n_neighbors=23)
neigh.fit(X_train_knn, y_train)
print(classification_report(y_test, neigh.predict(X_test_knn)))
accuracy_score(y_test, neigh.predict(X_test_knn))


# In[329]:


knn_cm = confusion_matrix(y_test, neigh.predict(X_test_knn))/len(y_test)
sns.heatmap(knn_cm, annot=True, cmap="YlGnBu", vmin=0)
plt.title("Confusion Matrix for KNN")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")


# In[108]:


# Find the best k
knn_acc_list = []
for i in range(1, 50): 
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train_knn, y_train)
    knn_pred = neigh.predict(X_val_knn)
    knn_acc = accuracy_score(y_val, knn_pred)
    knn_acc_list.append(knn_acc)


# In[127]:


# Drop irrelevant columns
X_train = X_train.drop(['Name', 'RescuerID', 'Description', 'PetID'], axis=1)
X_val = X_val.drop(['Name', 'RescuerID', 'Description', 'PetID'], axis=1)
X_test = X_test.drop(['Name', 'RescuerID', 'Description', 'PetID'], axis=1)


# In[139]:


# Logistic regression
# L2 Penalty
logreg = LogisticRegression(penalty='l2', random_state=205)
logreg.fit(X_train, y_train)


# In[140]:





# In[ ]:


classification_report


# In[141]:


logreg_l1 = LogisticRegression(penalty='l1', random_state=205)
logreg_l1.fit(X_train, y_train)
accuracy_score(y_val, logreg_l1.predict(X_val))


# In[308]:


print(classification_report(y_test, logreg_l1.predict(X_test)))
accuracy_score(y_test, logreg_l1.predict(X_test))


# In[258]:


# Random forest
rf = RandomForestClassifier(n_estimators=1200, max_depth=9)
rf.fit(X_train, y_train)


# In[259]:


accuracy_score(y_val, rf.predict(X_val))


# In[332]:


rf_cm = confusion_matrix(y_test, rf.predict(X_test))/len(y_test)
sns.heatmap(rf_cm, annot=True, cmap="YlGnBu", vmin=0)
plt.title("Confusion Matrix for Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")


# In[300]:


rf_importance = rf.feature_importances_


# In[301]:


indices = np.argsort(rf_importance)
features = np.array(['Cat or Dog', 'Age', 'First breed of pet', 'Secondary breed of pet',
                     'Gender', 'Color 1 of pet', 'Color 2 of pet', 
                     'Color 3 of pet','Size at maturity',
                     'Fur length', 'Vaccinated',  'Dewormed',  'Sterilized', 
           'Health Condition', 'Number of pets in profile',
            'Adoption Fee',   'State', 
           'Amount of Videos', 'Amount of Photos', 'Sentiment of Description'])
X_train.columns


# In[302]:


plt.figure(figsize=(8,6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), rf_importance[indices], color='orange', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')


# In[260]:


# Tune n_estimators and max_depth
# 1400, 8
rf_acc = []
for estimator_n in range(100, 2000, 100):
    for depth in range(3, 10):
        rf = RandomForestClassifier(n_estimators=estimator_n, max_depth=depth, random_state=205)
        rf.fit(X_train, y_train)
        rf_tuple = (estimator_n, depth, accuracy_score(y_val, rf.predict(X_val)))
        rf_acc.append(rf_tuple)


# In[311]:


rf = RandomForestClassifier(n_estimators=1400, max_depth=8, random_state=205)
rf.fit(X_train, y_train)


# In[312]:


print(classification_report(y_test, rf.predict(X_test)))
accuracy_score(y_test, rf.predict(X_test))


# In[261]:


# Tune n_estimators and max_depth
# 100, 8
gc_acc = []
for estimator_n in range(100, 1000, 100):
    print(estimator_n)
    for depth in range(5, 10):
        gc = GradientBoostingClassifier(n_estimators=estimator_n, max_depth=depth, random_state=205)
        gc.fit(X_train, y_train)
        gc_tuple = (estimator_n, depth, accuracy_score(y_val, gc.predict(X_val)))
        gc_acc.append(gc_tuple)


# In[313]:


gc = GradientBoostingClassifier(n_estimators=100, max_depth=8, random_state=205)
gc.fit(X_train, y_train)


# In[314]:


print(classification_report(y_test, gc.predict(X_test)))
accuracy_score(y_test, gc.predict(X_test))


# In[256]:


# Neural Net
mlp = MLPClassifier(hidden_layer_sizes= (10,10), max_iter=30, early_stopping=True, alpha=0.001)
mlp.fit(X_train, y_train)


# In[315]:


print(classification_report(y_test, mlp.predict(X_test)))
accuracy_score(y_test, mlp.predict(X_test))


# In[257]:


accuracy_score(y_val, mlp.predict(X_val))


# In[309]:


dt = DecisionTreeClassifier(max_depth=9)
dt.fit(X_train, y_train)


# In[266]:


accuracy_score(y_val, dt.predict(X_val))


# In[310]:


print(classification_report(y_test, dt.predict(X_test)))
accuracy_score(y_test, dt.predict(X_test))


# In[ ]:




