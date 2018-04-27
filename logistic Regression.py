
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_union


# In[3]:


comment_classes = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']


# In[4]:


train =  pd.read_csv('./train.csv').fillna(' ')


# In[5]:


test = pd.read_csv('./test.csv').fillna(' ')


# In[6]:


train_comments = train['comment_text']
test_comments = test['comment_text']


# In[7]:


all_comments = pd.concat([train_comments,test_comments])


# In[8]:


type(all_comments)


# # Convert a collection of raw documents to a matrix of TF-IDF features.

# In[9]:


word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=30000)


# In[10]:


# get_ipython().run_line_magic('pinfo', 'word_vectorizer')


# In[11]:


char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 4),
    max_features=30000)


# In[14]:


vectorizer = make_union(word_vectorizer, char_vectorizer, n_jobs=4)


# In[15]:


vectorizer.fit(all_comments)


# In[20]:


train_features = vectorizer.transform(train_comments)
test_features = vectorizer.transform(test_comments)


# In[16]:


scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})


# In[21]:


for class_name in comment_classes:
    train_target = train[class_name]
    classifier = LogisticRegression(solver='sag')

    cv_score = np.mean(cross_val_score(
        classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))

submission.to_csv('submission_logistic.csv', index=False)


# In[22]:



