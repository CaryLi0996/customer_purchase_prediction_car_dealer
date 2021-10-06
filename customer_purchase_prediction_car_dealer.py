#!/usr/bin/env python
# coding: utf-8

# ### Part 1 Data preparation and visualization
# 
# 1. Read the dataset 
# 2. Delete all the rows containing the missing data. Imputation is not necessary. 
# 3. Encode the data where necessary. 
# 4. Visualize the columns and their pairs. 
# 5. Produce the correlation matrix to make a first guess on usefulness of the predictors.

# In[2]:


data = pd.read_csv('dealer_data.csv'), usecols=['sex','age','income','purchase'])
data.head()


# In[3]:


data.shape


# In[4]:


data['sex'].value_counts()


# In[5]:


data[['age', 'income','purchase']].describe()


# In[6]:


for col in data.columns:

    missing_vals = data[data[col].isna()].shape[0]
    
    print("column", col, "has", missing_vals, "missing values")


# In[7]:


clean_data = data.dropna()


# In[8]:


for col in clean_data.columns:

    missing_vals = clean_data[clean_data[col].isna()].shape[0]
    
    print("column", col, "has", missing_vals, "missing values")


# In[9]:


clean_data['sex'] = pd.get_dummies(clean_data['sex'], drop_first=True)


# In[10]:


clean_data.corr()


# In[11]:


import seaborn as sns

g = sns.PairGrid(clean_data)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)


# ### Part 2 Inference by logistic regression
# 
# 1. You have 3 predictors (sex, age, income) and the target class variable purchase, taking the value 1 if he or she already bought a car once in the past. To predict this class for new potential customers, you need to learn the logistic regression model parameters. 
# 2. Divide your client base into training and testing sets. 
# 3. Fit the model on a training set. 
# 4. Produce your estimated regression equation, interpret the coefficients and comment on the regression summary. 
# 5. Try to reduce your model, dropping some predictors, and repeat the above steps with each reduced version. 
# 6. Compare the fitting results for your models.

# In[12]:


from sklearn.model_selection import train_test_split

X = clean_data.drop(['purchase'], axis=1)
y = clean_data['purchase']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=2)


# In[13]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=2, solver='newton-cg')
lr.fit(X_train, y_train)
coefs = lr.coef_
intercept = lr.intercept_


# In[14]:


sample = X_test.sample()
sample


# In[15]:


def regression_equation(features, coefs, intercept):
    
    z = intercept + np.dot(coefs, features.transpose())
    
    result = 1 / (1 + np.exp(-z))
    
    return 1-result, result


# In[16]:


res = regression_equation(sample, coefs, intercept)
res


# In[17]:


lr.predict_proba(sample)


# In[18]:


import itertools

features = ['sex', 'income', 'age']
feature_combinations = []

for i in range(1,len(features)+1):
   feature_combinations.extend(list(itertools.combinations(features,i)))

feature_combinations


# In[19]:


results = pd.DataFrame(columns=['model', 'features', 'coefficients', 'intercept'])

for features in feature_combinations:
    
    lr = LogisticRegression(random_state=2, solver='newton-cg', penalty='none')
    lr.fit(X_train[list(features)], y_train)
    coefs = lr.coef_
    intercept = lr.intercept_
    
    res = {'model': lr, 'features':features, 'coefficients':coefs, 'intercept': intercept}
    
    results = results.append(res, ignore_index=True)
    
    


# In[20]:


results


# ### Part 3. Prediction
# 
# 1. For each of the fitted models, predict the purchase class for the training set. 
# 2. For each of the fitted models, predict the purchase class for the testing set.

# In[21]:


predictions = pd.DataFrame(columns=['train', 'test'])

for i in results.index:
    pred_train = results['model'][i].predict(X_train[list(results['features'][i])])
    pred_test = results['model'][i].predict(X_test[list(results['features'][i])])
    
    result = {'train': pred_train, 'test':pred_test}
    
    predictions = predictions.append(result, ignore_index=True)
    
results = pd.concat([results,predictions],axis=1)
results


# ### Part 4. Evaluation of the prediction quality
# 
# 1. Produce the confusion matrix and the classification report for each of the predictions. 
# 2. Compare the results between the different models and also training versus testing sets.

# In[22]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score


# In[23]:


metrics = pd.DataFrame(columns=['train - confusion matrix', 'train - accuracy', 'train - precision', 'train - recall','train - f-score',
                               'test - confusion matrix', 'test - accuracy', 'test - precision', 'test - recall','test - f-score'])

for i in results.index:
    
    conf_train = confusion_matrix(y_train, results['train'][i])
    acc_train = accuracy_score(y_train, results['train'][i])
    prec_train = precision_score(y_train, results['train'][i])
    recall_train = recall_score(y_train, results['train'][i])
    f1_train = f1_score(y_train, results['train'][i])
    
    conf_test = confusion_matrix(y_test, results['test'][i])
    acc_test = accuracy_score(y_test, results['test'][i])
    prec_test = precision_score(y_test, results['test'][i])
    recall_test = recall_score(y_test, results['test'][i])
    f1_test = f1_score(y_test, results['test'][i])
    
    result = {'train - confusion matrix':conf_train, 
              'train - accuracy':acc_train, 
              'train - precision':prec_train, 
              'train - recall':recall_train,
              'train - f-score':f1_train,
              'test - confusion matrix':conf_test,
              'test - accuracy':acc_test,
              'test - precision':prec_test,
              'test - recall':recall_test,
              'test - f-score':f1_test}
    
    metrics = metrics.append(result, ignore_index = True)

results = pd.concat([results, metrics], axis=1)


# In[24]:


results

