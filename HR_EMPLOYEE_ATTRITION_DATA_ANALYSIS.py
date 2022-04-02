#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install category-encoders')


# # IMPORTING REQUIRED PACKAGES

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading Dataset

# In[6]:


data = pd.read_csv(r'C:\Users\ASUS\Downloads\HR_Employee_Attrition-1.csv')
data.head()
    


# In[7]:


data.shape


# In[8]:


data.columns


# In[9]:


data.isnull().sum()


# In[10]:


data.nunique()


# In[11]:


data.dtypes.sort_index()


# ## Looking For Duplicate Data

# In[12]:


data.duplicated().sum()


# In[13]:


data.describe()


# ## Data Analysis

# In[14]:


attrition_count = pd.DataFrame(data['Attrition'].value_counts())
attrition_count


# In[15]:


f, ax = plt.subplots(figsize=(10,10))
ax = data['Attrition'].value_counts(). plot.pie(explode=[0,0], autopct = '%1.1f%%', shadow=True)
ax.set_title('Attrition Probability')


# In[16]:


fig_dims = (12, 4)
fig, ax = plt.subplots(figsize=fig_dims)

#ax = axis
sns.countplot(
    x='Age', 
    hue='Attrition', 
    data = data, 
    palette="colorblind", 
    ax = ax,  
    edgecolor=sns.color_palette("dark", n_colors = 1),
    )


# In[17]:


f, ax = plt.subplots(2,2, figsize=(20,15))

ax[0,0] = sns.countplot(x='Attrition', hue= 'EducationField', data=data, ax = ax[0,0], palette='Set1' )
ax[0,0].set_title("Frequency Distribution of Attrition w.r.t. Education Field")

ax[1,0] = sns.countplot(x='Attrition', hue= 'Department', data=data,  ax = ax[1,0], palette='Set1' )
ax[1,0].set_title("Frequency Distribution of Attrition w.r.t. Department")

ax[0,1] = sns.countplot(x='Attrition', hue= 'Education', data=data,  ax = ax[0,1], palette='Set1' )
ax[0,1].set_title("Frequency Distribution of Attrition w.r.t. Education")

ax[1,1] = sns.countplot(x='Attrition', hue= 'BusinessTravel', data=data,  ax = ax[1,1], palette='Set1' )
ax[1,1].set_title("Frequency Distribution of Attrition w.r.t. Bussiness Travel")


f.tight_layout()


# In[18]:


f, ax = plt.subplots(2,2, figsize=(20,15))

ax[0,0] = sns.countplot(x='Attrition', hue= 'JobRole', data=data, ax = ax[0,0], palette='Set1' )
ax[0,0].set_title("Frequency Distribution of Attrition w.r.t. Job Role")

ax[0,1] = sns.countplot(x='Attrition', hue= 'OverTime', data=data,  ax = ax[0,1],palette='Set1' )
ax[0,1].set_title("Frequency Distribution of Attrition w.r.t. Over Time")

ax[1,1] = sns.countplot(x='Attrition', hue= 'EnvironmentSatisfaction', data=data,  ax = ax[1,1],palette='Set1' )
ax[1,1].set_title("Frequency Distribution of Attrition w.r.t. Environment Satisfaction")

ax[1,0] = sns.countplot(x='Attrition', hue='WorkLifeBalance', data=data, ax = ax[1,0], palette='Set1')
ax[1,0].set_title("Frequency Distribution of Attrition w.r.t. Work Life Balance")

f.tight_layout()


# ## Object data and theirs unique Values

# ## correlation of the columns

# In[19]:


for column in data.columns:
    if data[column].dtype == object:
        print(str(column) + ' : ' + str(data[column].unique()))
        print(data[column].value_counts())
        print("-"*90)


# In[20]:


#Remove unneeded columns

#Remove the column EmployeeNumber
data = data.drop('EmployeeNumber', axis = 1) # A number assignment 
#Remove the column StandardHours
data = data.drop('StandardHours', axis = 1) #Contains only value 80 
#Remove the column EmployeeCount
data = data.drop('EmployeeCount', axis = 1) #Contains only the value 1 
#Remove the column EmployeeCount
data = data.drop('Over18', axis = 1) #Contains only the value 'Yes'


# In[21]:


data.corr()


# In[22]:


plt.figure(figsize=(15,15))
sns.heatmap(
    data.corr(), 
    annot=True, 
    fmt='.0%',
    linewidths=1,
    cmap='inferno'
)


# ## Empty markdown cell, double click and press enter to edit

# In[ ]:


print(x.columns)


# In[ ]:


print(y)


# ## Train test split

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2,random_state=42 )


# In[ ]:


print(X_train.shape, X_test.shape)


# ## Converting controcial value to Numberical value

# In[ ]:


encoder = ce.OrdinalEncoder(cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.fit_transform(X_test)


# In[ ]:


Data preprocessing


# In[ ]:


cols = X_train.columns

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
X_train = pd.DataFrame(X_train, columns= [cols])
X_test = pd.DataFrame(X_test, columns=[cols])


# ## Builiding and testing model by Applying Random forest ML Algorithm

# In[ ]:


rfc = RandomForestClassifier(n_estimators=100, random_state=0)
rfc.fit(X_train, Y_train)
y_pred = rfc.predict(X_test)


# In[ ]:


score = accuracy_score(Y_test, y_pred)
print('randomforest classifier score: ', np.abs(score)*100


# In[ ]:


y_pred


# ## Final result

# ## Accuracy score=95.23

# In[ ]:




