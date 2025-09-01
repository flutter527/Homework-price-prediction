#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd


# In[30]:



import pandas as pd
# import pandas to load and process spreadsheet-type data

medical_dataset=pd.read_csv(r"E:\واجب نورا\data-example8.csv") 

# load a medical dataset.

medical_dataset


# In[ ]:





# In[31]:


set(medical_dataset['x1'])


# In[32]:



#SPLIT DB  


from sklearn.model_selection import train_test_split

# use the function to split the data, get 30% for testing and 70% for training.
train_data, test_data = train_test_split(medical_dataset, test_size=0.03, random_state=42)

#print the shapes (rows x columns) of the two datasets
print(train_data.shape)
print(test_data.shape)


# In[ ]:


#CREATE MODEL


from sklearn.tree import DecisionTreeClassifier

    
# create a Decision Tree Classifier
model=DecisionTreeClassifier(random_state=1)

# get the diagnosis column, to be used as the classification target
train_diagnoses=train_data['x1']

# drop the diagnosis column to get only the symptoms
train_patient_symptoms=train_data.drop(columns=['x1'])

# build a decision tree
model.fit(train_patient_symptoms, train_diagnoses)

# return the trained model
 


# In[ ]:



#TEST DATA


# functions used to evaluate a classifier
from sklearn.metrics import accuracy_score,confusion_matrix

# drop the diagnosis column to get only the symptoms
test_patient_symptoms=test_data.drop(columns=['x1'])

# get the diagnosis column, to be used as the classification target
test_diagnoses=test_data['x1']

# guess the most likely diagnoses
pred=model.predict(test_patient_symptoms)
print(pred)
#print the achieved accuracy score
accuracy_score(test_diagnoses,pred)


# In[ ]:


confusion_matrix(test_diagnoses,pred)


# In[ ]:




