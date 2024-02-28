#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Data Exploration

# In[2]:


df = pd.read_csv('Train_Data .csv')
print(df.shape)
df.head()


# In[3]:


df.info()


# In[4]:


cols = ['text', 'author', 'controversiality', 'parent_text', 'parent_score', 'parent_votes', 
        'parent_author', 'parent_controversiality', 'Score']
for col in cols:
    print(col,':',df[col].nunique())


# In[5]:


# Compare score with votes
df['score vs. votes'] = df['parent_score']==df['parent_votes']
df['score vs. votes'].nunique()


# In[6]:


# Since they are the same, we can drop one of them
df.drop(['parent_votes', 'score vs. votes'], axis= 1, inplace=True)
df.head()


# In[7]:


# Correlation of numerical features
cor = df.corr()
sns.heatmap(cor)


# # Natural Language Processing¶
# 

# In[8]:


# Transfer category values to be lowercased & remove leading and trailing whitespaces
categorical_cols = ['text','author','parent_text','parent_author']
for col in df[categorical_cols]:
    df[col] = df[col].str.lower()
    df[col] = df[col].str.strip()
df.head()


# In[9]:


#Remove punctuation marks
import string

for col in df[categorical_cols]:
    df[col] = df[col].apply(lambda x:''.join([i for i in x if i not in string.punctuation]))
df.head()


# In[10]:


#tokenization


# In[11]:



import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize 

def text_tokens(row):
    text = row['text']
    tokens = word_tokenize(text)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words
df['text_tokens'] = df.apply(text_tokens, axis=1)

def parent_text_tokens(row):
    parent_text = row['parent_text']
    tokens = word_tokenize(parent_text)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words
df['parent_text_tokens'] = df.apply(parent_text_tokens, axis=1)

df.head()


# In[12]:


#stop word removal
stop_words = stopwords.words('english')

tokens_cols = ['text_tokens','parent_text_tokens']

for col in tokens_cols:
    df[col] = df[col].apply(lambda x: ' '.join([w for w in x if w not in (stop_words)]))
df.head()


# In[13]:


# Vectorize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = CountVectorizer()
text = vectorizer.fit_transform(df['text']).toarray()
text = pd.DataFrame(text, columns=vectorizer.get_feature_names())

text.shape


# In[14]:


vectorizer1 = TfidfVectorizer(max_features=50,min_df=1,max_df=0.7)
text_tf_idf = vectorizer1.fit_transform(df['text']).toarray()
text_tf_idf = pd.DataFrame(text_tf_idf, columns=vectorizer1.get_feature_names())

text_tf_idf.shape


# In[15]:


num_cols = df[['controversiality', 'parent_score', 'parent_controversiality']]
x = pd.concat([text_tf_idf, num_cols], axis=1)
y = df['Score']


# In[16]:


# import train_test_split
from sklearn.model_selection import train_test_split

# split the data
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2, random_state = 42)


# #    Machine Learining Models

# In[17]:


# Linear regressor
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

pred_y = lr.predict(x_val)


# In[18]:


# Root mean squared error 
from sklearn.metrics import mean_squared_error
print('Root Mean Squared Error is: ', np.sqrt(mean_squared_error(y_val, pred_y)))


# In[19]:


# KNN 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

# Hyperparameter for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 100)}
knn = KNeighborsRegressor()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(x_train, y_train)
knn_cv.best_params_


# In[20]:


knn = KNeighborsRegressor(n_neighbors = 50)
knn.fit(x_train, y_train)

pred_y2 = knn.predict(x_val)

# Root mean squared error 
print('Root Mean Squared Error is: ', np.sqrt(mean_squared_error(y_val, pred_y2)))


# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
param_grid = {'n_estimators': np.arange(1, 50), 'max_depth': np.arange(1, 50)}

RFRegressor = RandomForestRegressor()
RFRegressor_cv = GridSearchCV(RFRegressor, param_grid, cv=5)
RFRegressor_cv.fit(x_train, y_train)
RFRegressor_cv.best_params_


# In[ ]:


randForest = RandomForestRegressor(n_estimators=9, max_depth=3, max_features='auto')
randForest.fit(x_train, y_train)

pred_y3 = randForest.predict(x_val)

# Root mean squared error 
print('Root Mean Squared Error is: ', np.sqrt(mean_squared_error(y_val, pred_y3)))


# # Test Data

# In[ ]:


test_data = pd.read_csv('Test_Data.csv')
test_data.head()


# In[ ]:


# Natural Laguage Preprocessing

test_data['text'] = test_data['text'].str.lower()
test_data['text'] = test_data['text'].str.strip()

test_data['text'] = test_data['text'].apply(lambda x:''.join([i for i in x if i not in string.punctuation]))

test_data['text_tokens'] = df.apply(text_tokens, axis=1)

test_data['text_tokens'] = test_data['text_tokens'].apply(lambda x: ' '.join([w for w in x if w not in (stop_words)]))

test_data['text_tokens'] = test_data['text_tokens'].apply(lemmatize_function)

test_data['text']= test_data['text_tokens'].apply(lambda x: ' '.join(x))
test_data.drop(['text_tokens'], axis=1, inplace= True)

text_data_tf_idf = vectorizer1.fit_transform(test_data['text']).toarray()
text_data_tf_idf = pd.DataFrame(text_data_tf_idf, columns=vectorizer1.get_feature_names())


# In[ ]:


# Select features of test data
test_num = test_data[['controversiality','parent_score', 'parent_controversiality']]
#num_cols = df[['controversiality', 'parent_score', 'parent_controversiality']]
test = pd.concat([text_data_tf_idf, test_num], axis=1)


# In[ ]:


# Predict score
predict_test_y = lr.predict(test)
predict_test_y1 = xgbReg.predict(test)
predict_test_y2 = knn.predict(test)
predict_test_y3 = randForest.predict(test)
