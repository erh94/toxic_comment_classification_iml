import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import roc_auc_score
from keras.models import model_from_json
from keras.models import Model
from keras.layers import Input,Dense,Embedding,SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback

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

def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(7, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model




if __name__ == "__main__":

	# Read inputs a) Max Feat b) Embed_size c) Embed_matrix d) maxlen 

	model = get_model()
	print("Created Model \n")


	batch_size = 32
	epochs = 2

	# Read x_train_1 y_train

	X_tra, X_val, y_tra, y_val = train_test_split(x_train_1, y_train, train_size=0.95, random_state=233)
	RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

	hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                 callbacks=[RocAuc], verbose=2)

	# serialize model to JSON
	model_json = model.to_json()
	with open("./models/GRU_feat_none.json", "w") as json_file:
    	json_file.write(model_json)

	# serialize weights to HDF5
	model.save_weights("./models/GRU_feat_none.h5")
	print("Saved model to disk")