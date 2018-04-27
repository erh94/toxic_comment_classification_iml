
# coding: utf-8



import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model,Sequential
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import model_from_json
from keras.regularizers import l2,l1

import datetime

now = datetime.datetime.now()
suffix = now.strftime("%Y%m%d%H%M")
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')
subm = pd.read_csv('./input/sample_submission.csv')
emb_file = './embeddings/glove.6B.300d.txt'

# In[123]:


train.isnull().any(),test.isnull().any()


# In[124]:


list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train['none'] = 1-train[list_classes].max(axis=1)
list_classes.append('none')
y = train[list_classes].values
list_sentences_train = train["comment_text"]
list_sentences_test = test["comment_text"]


# In[125]:


max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train)+list(list_sentences_test))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
word_index = tokenizer.word_index


# In[126]:


list_tokenized_train[:1]


# In[127]:


maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


# In[128]:


totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]
max(totalNumWords)


# In[129]:


plt.hist(totalNumWords,bins = np.arange(0,410,10))
#[0,50,100,150,200,250,300,350,400])#,450,500,550,600,650,700,750,800,850,900])
plt.savefig('histogram.png')


# In[130]:


inp = Input(shape=(maxlen, ))


# In[131]:


embeddings_index = {}
f = open(emb_file)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[132]:


embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[133]:


embed_size = 128
#x = Embedding(len(word_index)+1,50,weights=[embedding_matrix],input_length=embed_size,trainable=False)(inp)
model = Sequential()
model.add(Embedding(max_features, embed_size))
#x = Embedding(max_features, embed_size)(inp)


# In[134]:


#x = LSTM(60, return_sequences=True,name='lstm_layer',activation='sigmoid')(x)
model.add(LSTM(60, return_sequences=True,name='lstm_layer',activation='tanh'))


# In[135]:


#x = GlobalMaxPool1D()(x)
model.add(GlobalMaxPool1D())


# In[136]:


#x = Dropout(0.2)(x)
model.add(Dropout(0.2))


# In[137]:


#x = Dense(50, activation="relu")(x)
model.add(Dense(50, activation="relu"))


# In[138]:


#x = Dropout(0.2)(x)
model.add(Dropout(0.2))


# In[139]:


#x = Dense(7, activation="sigmoid",activity_regularizer=l2(0.0001))(x)
model.add(Dense(7, activation="sigmoid"))


# In[140]:


#model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


# In[141]:


batch_size = 32
epochs = 2
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# In[142]:


model.summary()


# In[143]:


model_json = model.to_json()
with open("model_LSTM"+suffix+".json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_LSTM"+suffix+".h5")
print("Saved model to disk")


# In[144]:


json_file = open("model_LSTM"+suffix+".json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_LSTM"+suffix+".h5")
print("Loaded model from disk")


# In[145]:


y_pred = loaded_model.predict(X_te, batch_size=1024)
y_pred = np.delete(y_pred,6,1)
subm[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
subm.to_csv('submission_LSTM_v1'+suffix+'.csv', index=False)

