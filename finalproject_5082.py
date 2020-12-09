#!/usr/bin/env python
# coding: utf-8

# In[183]:


import pandas as pd 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style='whitegrid')

# Modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import sklearn.metrics as metrics
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')


# In[184]:


import warnings
warnings.filterwarnings("ignore")


# In[185]:


import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


# In[186]:


data_train.head()


# In[187]:


data_train.isna().sum()


# In[188]:


data_test.isna().sum()


# In[189]:


from matplotlib import pyplot as plt 
plt.bar(
    x = ['Not interested', 'Is interested'], 
    height = [data_train.Response.value_counts()[0], data_train.Response.value_counts()[1]], 
    color = ['blue','green']
 ); 
plt.title("How many people is interested in getting car insurance? ")
plt.ylabel('Total ammount of customers')
plt.show()


# In[190]:


# Filtering only those who took an insurance
takers_index = data_train.Response == 1
takers = data_train[takers_index]

# Customers who are not taking car insurance
not_index = data_train.Response == 0 
not_takers = data_train[not_index]


# In[191]:



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Customers separated by sex")

ax1.bar(
    x = ['Male', 'Female'], 
    height = [takers.Gender.value_counts()[0], takers.Gender.value_counts()[1]], 
    color = ['yellow', 'pink']
); 
ax1.set_title("Want to take car insurance")
ax1.set_ylabel("Total ammount of customers")

ax2.bar(
    x = ['Male', 'Female'], 
    height = [not_takers.Gender.value_counts()[0], not_takers.Gender.value_counts()[1]], 
    color = ['yellow', 'pink']
); 
ax2.set_title("Do not want to take car insurance")
ax2.set_ylabel("Total ammount of customers")


# In[192]:


pd.crosstab(data_train['Response'], data_train['Gender']).plot(kind="bar", figsize=(10,6))

plt.title("Response distribution for Gender")
plt.xlabel("0 = Customer is Not interested, 1 = Customer is interested")
plt.ylabel("Amount")
plt.legend(["Female", "Male"])
plt.xticks(rotation=0);


# In[193]:


#ditribution of Gender,Driving_License,Previously_Insured,Previously_Insured
fig, axarr = plt.subplots(2, 2, figsize=(10, 10))

data_train['Gender'].value_counts().sort_index().plot.pie(
    ax=axarr[0][0])
axarr[0][0].set_title("Gender", fontsize=18)
data_train['Previously_Insured'].value_counts().sort_index().plot.pie(
    ax=axarr[1][0])
axarr[1][0].set_title("Previously_Insured", fontsize=18)

data_train['Vehicle_Damage'].value_counts().sort_index().plot.pie(
    ax=axarr[1][1])
axarr[1][1].set_title("Vehicle_Damage", fontsize=18)

data_train['Driving_License'].value_counts().head().plot.pie(
    ax=axarr[0][1])
axarr[0][1].set_title("Driving_License", fontsize=18)


# In[194]:


# Age plot 

plt.title("Customers ages (who want to take car insurance)")
plt.xlabel("Age")
plt.ylabel("Total ammount of customers")
takers.Age.hist()
plt.show()

# As we can see, most of the customers who take car insurance are aged between 30 and 50


# In[195]:


data_train['Previously_Insured'].value_counts()


# In[196]:


pd.crosstab(data_train['Response'], data_train['Previously_Insured'])


# In[197]:


pd.crosstab(data_train['Response'], data_train['Previously_Insured']).plot(kind="bar", figsize=(10,6))

plt.title("Response distribution for Previously_Insured")
plt.xlabel("0 = Customer is Not interested, 1 = Customer is interested")
plt.ylabel("Amount")
plt.color=("blue","green")
plt.legend(["Customer doesn't have Vehicle Insurance", "Customer already has Vehicle Insurance"])
plt.xticks(rotation=0);


# In[198]:


data_train['Vehicle_Age'].value_counts()


# In[199]:


pd.crosstab(data_train['Response'], data_train['Vehicle_Age']).plot(kind="bar", figsize=(10,6))

plt.title("Response distribution for Vehicle_Age")
plt.xlabel("0 = Customer is Not interested, 1 = Customer is interested")
plt.ylabel("Amount")
plt.legend(["1-2 Year", "< 1 Year", "> 2 Years"])
plt.xticks(rotation=0);


# In[200]:


data_train['Vehicle_Damage'].value_counts()


# In[201]:


pd.crosstab(data_train['Response'], data_train['Vehicle_Damage']).plot(kind="bar", figsize=(10,6))

plt.title("Response distribution for Vehicle_Damage")
plt.xlabel("0 = Customer is Not interested, 1 = Customer is interested")
plt.ylabel("Amount")
plt.legend(["Vehicle Damage", "No Vehicle Damage"])
plt.xticks(rotation=0);


# In[202]:


A= sns.boxplot(y='Annual_Premium', x='Response', data=data_train)
A.set_title("Annual_Premium Distribution for each Response");


# In[203]:


data_train['Policy_Sales_Channel'].describe()


# In[204]:


A= sns.boxplot(y='Policy_Sales_Channel', x='Response', data=data_train);
A.set_title("Policy_Sales_Channel Distribution for each Response");


# In[205]:


data_train['Vintage'].describe()


# In[206]:


A= sns.boxplot(y='Vintage', x='Response', data=data_train);
A.set_title("Vintage Distribution for each Response");


# In[207]:


data_train['Premium_Per_Day'] = (data_train.Annual_Premium / 365) * data_train.Vintage
data_test['Premium_Per_Day'] = (data_test.Annual_Premium / 365) * data_test.Vintage

data_train.head()


# In[208]:


# Visualize the premium that customers pays 

plt.figure(figsize = (20, 8))
plt.plot(
    data_train.id, data_train.Annual_Premium, 
    label = "Anual premium", 
)
plt.plot(
    data_train.id, data_train.Premium_Per_Day, 
    label = "Daily premium"
)
plt.legend()
plt.title("Anual and daily premium (200 customers)")
plt.show()


# In[209]:


data_train = pd.concat([data_train[['Age', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage', 'Response']],
           pd.get_dummies(data_train[['Gender', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage']])], axis=1)


# In[210]:


data_train.head()


# In[211]:


plt.figure(figsize=(12,10))
cor = data_train.corr()
sns.heatmap(cor, annot=True)
plt.show()


# In[230]:


data_train.head(10)


# In[215]:


y=data_train.Response
X=data_train.drop(columns=['Response'])


# In[216]:


## split into 70%train set and 30%test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[218]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_Predict = rf.predict(X_test)


# In[219]:


print(classification_report(y_test, rf_Predict))
rf_accuracy = accuracy_score(y_test, rf_Predict)
print("Accuracy of randomforest" + ' : ' + str(rf_accuracy))


# In[220]:


cv_scores = cross_val_score(rf,X,y,cv=10)

print(cv_scores)
print("Average 10-Fold CV Score: {}".format(np.mean(cv_scores)))


# In[221]:


# Plot ROC_AUC for random forest
probs = rf.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

#  plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic for Random Forest')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[222]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_predict = lr.predict(X_test)


# In[223]:


print(classification_report(y_test, lr_predict))
lr_accuracy = accuracy_score(y_test, lr_predict)
print("Accuracy of Logistic Regression" + ' : ' + str(lr_accuracy))


# In[224]:


#Plot ROC_AUC for logistic regression
probs = lr.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic for Logistic Regression')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[228]:


######KNN


# In[225]:


# build the knn model and calculate the accuracy score when n=10
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn_predict = knn.predict(X_test)


# In[226]:


knn_accuracy = accuracy_score(y_test, knn_predict)
print("Accuracy of Logistic Regression" + ' : ' + str(knn_accuracy))


# In[227]:


# Plot ROC_AUC for knn
probs = knn.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

#  plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic for KNN')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:




