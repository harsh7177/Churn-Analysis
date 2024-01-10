#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data= pd.read_csv(r'Telco_Churn_Data.csv')
data.head(5)


# In[3]:


len(data)


# In[4]:


data.shape


# In[5]:


data.isnull().values.any()


# In[6]:


data.info()


# In[7]:


data.columns=data.columns.str.replace(' ','_')


# In[8]:


data['Avg_Hours_WorkOrderOpened'] = data['Avg_Hours_WorkOrderOpenned']
data.drop(["Avg_Hours_WorkOrderOpenned"],axis=1,inplace=True)


# In[9]:


data.columns


# In[10]:


data.describe()


# In[11]:


data.describe(include='object')


# In[12]:


data['Target_Code']=data.Target_Code.astype('object') 
data['Condition_of_Current_Handset']=data.Condition_of_Current_Handset.astype('object') 
data['Current_TechSupComplaints']=data.Current_TechSupComplaints.astype('object') 
data['Target_Code']=data.Target_Code.astype('int64') 


# In[13]:


data.describe(include='object')


# In[14]:


round(data.isnull().sum()/len(data)*100,2)


# In[15]:


data.Complaint_Code.value_counts()


# In[16]:


data.Condition_of_Current_Handset.value_counts()


# In[17]:


data['Complaint_Code']=data['Complaint_Code'].fillna(value='Billing Problem') 
data['Condition_of_Current_Handset']=data['Condition_of_Current_Handset'].fillna(value=1) 
data['Condition_of_Current_Handset']=data.Condition_of_Current_Handset.astype('object') 


# In[18]:


data['Target_Churn'].value_counts(0)


# In[19]:


data['Target_Churn'].value_counts(1)*100


# In[20]:


summary_churn = data.groupby('Target_Churn')
summary_churn.mean()


# In[21]:


corr = data.corr() 
plt.figure(figsize=(15,8)) 
sns.heatmap(corr, 
            xticklabels=corr.columns.values, 
            yticklabels=corr.columns.values,annot=True,cmap='Greys_r') 
corr 


# ### Univariate Analysis

# In[22]:


f, axes = plt.subplots(ncols=3, figsize=(15, 6)) 
sns.distplot(data.Avg_Calls_Weekdays, kde=True, \
             color="gray", \
             ax=axes[0]).set_title('Avg_Calls_Weekdays') 
axes[0].set_ylabel('No of Customers') 
sns.distplot(data.Avg_Calls, kde=True, color="gray", \
             ax=axes[1]).set_title('Avg_Calls') 
axes[1].set_ylabel('No of Customers') 
sns.distplot(data.Current_Bill_Amt, kde=True, color="gray", \
             ax=axes[2]).set_title('Current_Bill_Amt') 
axes[2].set_ylabel('No of Customers') 


# ### Bivariate Analysis

# In[23]:


plt.figure(figsize=(17,10)) 
p=sns.countplot(y="Complaint_Code", hue='Target_Churn', \
                data=data,palette="Greys_r") 
legend = p.get_legend() 
legend_txt = legend.texts 
legend_txt[0].set_text("No Churn") 
legend_txt[1].set_text("Churn") 
p.set_title('Customer Complaint Code Distribution') 


# In[24]:


plt.figure(figsize=(15,4)) 
p=sns.countplot(y="Acct_Plan_Subtype", hue='Target_Churn', \
                data=data,palette="Greys_r") 
legend = p.get_legend() 
legend_txt = legend.texts 
legend_txt[0].set_text("No Churn") 
legend_txt[1].set_text("Churn") 
p.set_title('Customer Acct_Plan_Subtype Distribution') 


# In[25]:


plt.figure(figsize=(15,4)) 
p=sns.countplot(y="Current_TechSupComplaints", hue='Target_Churn', \
                data=data,palette="Greys_r") 
legend = p.get_legend() 
legend_txt = legend.texts 
legend_txt[0].set_text("No Churn") 
legend_txt[1].set_text("Churn") 
p.set_title('Customer Current_TechSupComplaints Distribution') 


# In[26]:


plt.figure(figsize=(15,4)) 
ax=sns.kdeplot(data.loc[(data['Target_Code'] == 0), \
                        'Avg_Days_Delinquent'] , \
               color=sns.color_palette("Greys_r")[0],\
               shade=True,label='no churn') 
ax=sns.kdeplot(data.loc[(data['Target_Code'] == 1),\
                        'Avg_Days_Delinquent'] , \
               color=sns.color_palette("Greys_r")[1],\
               shade=True, label='churn') 
ax.set(xlabel='Average No of Days Deliquent/Defalulted \
from paying', ylabel='Frequency') 
plt.title('Average No of Days Deliquent/Defalulted from \
paying - churn vs no churn')
plt.legend()


# In[27]:


plt.figure(figsize=(15,4)) 
ax=sns.kdeplot(data.loc[(data['Target_Code'] == 0), \
                        'Account_Age'], \
               color=sns.color_palette("Greys_r")[0], \
               shade=True,label='no churn') 
ax=sns.kdeplot(data.loc[(data['Target_Code'] == 1), \
                        'Account_Age'], \
               color=sns.color_palette("Greys_r")[1] ,\
               shade=True, label='churn') 
ax.set(xlabel='Account_Age', ylabel='Frequency') 
plt.title('Account_Age - churn vs no churn') 
plt.legend()


# In[28]:


plt.figure(figsize=(15,4)) 
ax=sns.kdeplot(data.loc[(data['Target_Code'] == 0), \
                        'Percent_Increase_MOM'], \
               color=sns.color_palette("Greys_r")[0], \
               shade=True, label='no churn') 
ax=sns.kdeplot(data.loc[(data['Target_Code'] == 1), \
                        'Percent_Increase_MOM'], \
               color=sns.color_palette("Greys_r")[1], \
               shade=True, label='churn') 
ax.set(xlabel='Percent_Increase_MOM', ylabel='Frequency') 
plt.title('Percent_Increase_MOM- churn vs no churn')
plt.legend()


# ### Feature Selection

# In[29]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np


# In[30]:


data["Acct_Plan_Subtype"] = data["Acct_Plan_Subtype"]\
.astype('category').cat.codes 
data["Complaint_Code"] = data["Complaint_Code"]\
.astype('category').cat.codes 


# In[31]:


data[["Acct_Plan_Subtype","Complaint_Code"]].head()


# In[32]:


target = 'Target_Code' 
X = data.drop(['Target_Code','Target_Churn'], axis=1) 
y = data[target] 
X_train, X_test, y_train, y_test = train_test_split\
(X, y, test_size=0.15, \
 random_state=123, stratify=y)


# In[33]:


forest=RandomForestClassifier(n_estimators=500,random_state=1) 
forest.fit(X_train,y_train) 

importances=forest.feature_importances_ 
features = data.drop(['Target_Code','Target_Churn'],axis=1)\
.columns 
indices = np.argsort(importances)[::-1] 

plt.figure(figsize=(15,4)) 
plt.title("Feature importances using Random Forest") 
plt.bar(range(X_train.shape[1]), importances[indices],\
        color="gray", align="center") 
plt.xticks(range(X_train.shape[1]), features[indices], \
           rotation='vertical', fontsize=15) 
plt.xlim([-1, X_train.shape[1]])
plt.show()


# ### Logistic Regression

# In[34]:


import statsmodels.api as sm 
top7_features = ['Avg_Days_Delinquent','Percent_Increase_MOM',\
                 'Avg_Calls_Weekdays','Current_Bill_Amt',\
                 'Avg_Calls','Complaint_Code','Account_Age'] 
logReg = sm.Logit(y_train, X_train[top7_features]) 
logistic_regression = logReg.fit() 


# In[35]:


logistic_regression.summary
logistic_regression.params


# In[36]:


coef = logistic_regression.params 

def y (coef, Avg_Days_Delinquent, Percent_Increase_MOM, \
       Avg_Calls_Weekdays, Current_Bill_Amt, Avg_Calls, \
       Complaint_Code, Account_Age): 
    
    final_coef=coef[0]\
    *Avg_Days_Delinquent\
    +coef[1]*Percent_Increase_MOM\
    +coef[2]\
    *Avg_Calls_Weekdays\
    +coef[3]*Current_Bill_Amt\
    +coef[4]*Avg_Calls+coef[5]\
    *Complaint_Code+coef[6]\
    *Account_Age 

    return final_coef 


# In[37]:


import numpy as np

y1 = y(coef, 40, 5, 39000,12000,9000,0,17)
p = np.exp(y1) / (1+np.exp(y1))
p

