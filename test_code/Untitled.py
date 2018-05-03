
# coding: utf-8

# In[1]:


import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import model_from_json
from keras.regularizers import l2,l1


# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
subm = pd.read_csv('sample_submission.csv')


# In[3]:


train.isnull().any(),test.isnull().any()


# In[31]:


list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_train = train["comment_text"]
list_sentences_test = test["comment_text"]


# In[32]:


max_features = 30000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
word_index = tokenizer.word_index


# In[33]:


list_tokenized_train[1]


# In[34]:


maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


# In[35]:


totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]


# In[36]:


totalNumWords_5=0
for size in totalNumWords:
    if(size==100):
        totalNumWords_5+=1

totalNumWords_5


# In[37]:


plt.hist(totalNumWords,bins = np.arange(0,500,10))#[0,50,100,150,200,250,300,350,400])#,450,500,550,600,650,700,750,800,850,900])
plt.show()


# In[38]:


inp = Input(shape=(maxlen, ))


# In[159]:


embeddings_index = {}
f = open(os.path.join('glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[163]:


embedding_matrix = np.zeros((len(word_index) + 1, 50))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[164]:


embed_size = 128
x = Embedding(len(word_index)+1,50,weights=[embedding_matrix],input_length=embed_size,trainable=False)(inp)


# In[165]:


x = LSTM(60, return_sequences=True,name='lstm_layer',activation='sigmoid')(x)


# In[166]:


x = GlobalMaxPool1D()(x)


# In[167]:


x = Dropout(0.1)(x)


# In[168]:


x = Dense(50, activation="relu")(x)


# In[169]:


x = Dropout(0.1)(x)


# In[170]:


x = Dense(6, activation="sigmoid",activity_regularizer=l2(0.0001))(x)


# In[171]:


model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


# In[172]:


batch_size = 32
epochs = 2
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# In[173]:


model.summary()


# In[174]:


model_json = model.to_json()
with open("model_LSTM.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_LSTM.h5")
print("Saved model to disk")


# In[175]:


json_file = open('model_LSTM.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_LSTM.h5")
print("Loaded model from disk")


# In[176]:


y_pred = loaded_model.predict(X_te, batch_size=1024)
subm[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
subm.to_csv('submission_LSTM_v1.csv', index=False)

