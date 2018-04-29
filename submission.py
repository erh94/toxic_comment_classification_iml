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

if __name__ == "__main__":

	submission = pd.read_csv('./input/sample_submission.csv')
	list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
	json_file = open('./models/GRU_feat_none.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("./models/GRU_feat_none.h5")
	print("Loaded model from disk")

	with open('./models/x_test.pkl','rb') as handle: 
		x_test = pickle.load(handle)

	y_pred = loaded_model.predict(x_test, batch_size=1024)
	y_pred = np.delete(y_pred,6,1)
	submission[list_classes] = y_pred
	#submission.drop(['none'],axis=1)
	submission.to_csv('./output/submission_GRU_glove.csv', index=False)
