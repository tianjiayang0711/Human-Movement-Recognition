
# coding: utf-8

# In[ ]:


# Intro：


# In[1]:


import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.grid_search import GridSearchCV


# In[3]:



xtrain=pd.read_table('C:/Human_Activity_Recognition/dataset/train/X_train.txt',delim_whitespace=True,header=None)
xtest=pd.read_table('C:/Human_Activity_Recognition/dataset/test/X_test.txt',delim_whitespace=True,header=None)
ytrain=pd.read_table('C:/Human_Activity_Recognition/dataset/train/y_train.txt',header=None)
ytest=pd.read_table('C:/Human_Activity_Recognition/dataset/test/y_test.txt',header=None)


# In[7]:


classifier=svm.SVC()


# In[8]:


parameters=[{'kernel': ['rbf'], 'gamma': [1,0,1,0.01,0.001, 0.0001], 'C': [0.1,1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


# In[10]:


model=GridSearchCV(classifier,parameters,n_jobs=6,cv=5,verbose=4)
model.fit(xtrain.as_matrix(),ytrain.as_matrix().ravel().T)


# In[11]:


print(model.best_params_)


# Further more turing on C:

# In[16]:


clf = svm.SVC(gamma=0.01,kernel='rbf',cache_size=1000)
parametersc = {'C':[500,600,700,800,900,1000,1100,1200,1300,1400,1500]}
model=GridSearchCV(clf,parametersc,n_jobs=6,cv=5,verbose=4)
model.fit(xtrain.as_matrix(),ytrain.as_matrix().ravel().T)


# In[17]:


print(model.best_params_)


# In[18]:


clf = svm.SVC(gamma=0.01,kernel='rbf',cache_size=2000)
parametersc = {'C':[150,200,250,350,400,450]}
model=GridSearchCV(clf,parametersc,n_jobs=6,cv=5,verbose=4)
model.fit(xtrain.as_matrix(),ytrain.as_matrix().ravel().T)


# In[19]:


print(model.best_params_)


# In[20]:


clf = svm.SVC(gamma=0.01,kernel='rbf',cache_size=2000)
parametersc = {'C':[110,120,130,140,150,160,170,180,190]}
model=GridSearchCV(clf,parametersc,n_jobs=6,cv=5,verbose=4)
model.fit(xtrain.as_matrix(),ytrain.as_matrix().ravel().T)


# In[21]:


print(model.best_params_)


# Now,let's train our final SVM model

# In[24]:


model_best = svm.SVC(kernel='rbf',gamma=0.01,C=110,cache_size=5000)
model_best.fit(xtrain.as_matrix(),ytrain.as_matrix().ravel().T)


# Performance on test set

# In[25]:


from sklearn.metrics import accuracy_score
ypred=model_best.predict(xtest)
accuracy=accuracy_score(ytest,ypred)
print ('Best Parameters: '+ str(model.best_params_))
print ('Accuracy Score: '+ str(accuracy*100) + ' %')


# confusion matrix on test set

# In[27]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
confusion = confusion_matrix(ytest, ypred)
plt.figure(figsize=(10, 7))
sns.heatmap(confusion, annot=True)
plt.show()


# As we can see, there is a little bit problem to classify between sitting and standing.
# <br>
# 0 WALKING
# 1 WALKING_UPSTAIRS
# 2 WALKING_DOWNSTAIRS
# 3 SITTING
# 4 STANDING
# 5 LAYING

# As we can see from the result, SVM with RBF kernal is a strong weapon for these data and features.
# 
# And the SVM's performance is nearly equal to the LR model 
# 
# Time to review the math, Let's compare those two models
