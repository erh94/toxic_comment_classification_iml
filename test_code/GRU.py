
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# In[2]:


from keras.models import model_from_json


# In[3]:


from keras.models import Model
from keras.layers import Input,Dense,Embedding,SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback


# In[4]:


import os
os.environ['OMP_NUM_THREADS']='4'


# In[5]:


Embedding_file = './glove.6B.50d.txt'


# In[6]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')


# In[20]:


# len(train["comment_text"][1].split())
train["comment_text"][1]
# len("hi".split())
# train["comment_text"][2]
# train.head()["comment_text"]


# In[21]:


# train["comment_text"]
x_train = train["comment_text"].fillna("fillna").values
# X_train is an array
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
# y_train
x_test = test["comment_text"].fillna("fillna").values


# In[22]:


x_test


# # Embeddings and features extraction

# In[23]:


max_features = 30000
maxlen = 400
embed_size = 300


# In[24]:


tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x_train) + list(x_test))
X_train_1 = tokenizer.texts_to_sequences(x_train)
X_test_1 = tokenizer.texts_to_sequences(x_test)


# In[25]:


x_train_1 = sequence.pad_sequences(X_train_1, maxlen=maxlen)
x_test_1 = sequence.pad_sequences(X_test_1, maxlen=maxlen)


# In[28]:


# len(X_train[0])
len(X_train_1[1])


# In[30]:


X_train_1[1]


# In[35]:


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


# In[36]:


get_coefs(1,[1,2,3])


# In[37]:


embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(Embedding_file))


# In[38]:


word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[39]:


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))


# In[40]:


def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


# In[41]:


model = get_model()


# In[ ]:


batch_size = 32
epochs = 2

X_tra, X_val, y_tra, y_val = train_test_split(x_train_1, y_train, train_size=0.95, random_state=233)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                 callbacks=[RocAuc], verbose=2)


# In[ ]:


# serialize model to JSON
model_json = model.to_json()
with open("model_GRU.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_GRU.h5")
print("Saved model to disk")


# In[ ]:


get_ipython().system('ls')


# In[ ]:


# load json and create model
json_file = open('model_GRU.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_GRU.h5")
print("Loaded model from disk")


# In[ ]:


y_pred = loaded_model.predict(x_test, batch_size=1024)
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('submission_GRU_glove.csv', index=False)

