import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# In[3]:


from keras.models import model_from_json


# In[4]:


from keras.models import Model
from keras.layers import Input,Dense,Embedding,SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train['none'] = 1-train[list_classes].max(axis=1)
list_classes.append('none')

print(list_classes)
x_train = train["comment_text"].fillna("fillna").values
# X_train is an array
y_train = train[list_classes].values
print(y_train)
x_test = test["comment_text"].fillna("fillna").values

max_features = 30000
maxlen = 100
embed_size = 300
print("feature_size {0} MaxLen of sequences {1} Embedding_size {2}".format(max_features , maxlen , embed_size))


# In[11]:


tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x_train) + list(x_test))
X_train_1 = tokenizer.texts_to_sequences(x_train)
X_test_1 = tokenizer.texts_to_sequences(x_test)


# In[12]:

#padded sequences used for prediction and training 
x_train_1 = sequence.pad_sequences(X_train_1, maxlen=maxlen)
x_test_1 = sequence.pad_sequences(X_test_1, maxlen=maxlen)

json_file = open('GRU_feat_none_300.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("GRU_feat_none_300.h5")
print("Loaded model from disk")



y_pred = loaded_model.predict(x_test_1, batch_size=1024)
print(y_pred.shape)
y_del=np.delete(y_pred,6,1)
print(y_del.shape)
print(list_classes)
list_classes.remove("none")
print(submission.shape)
# print(list_classes)
submission[list_classes] = y_del
# submission.drop(['none'],axis=1)
submission.to_csv('300_feat_submission_GRU_glove.csv', index=False)