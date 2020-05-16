import numpy as np 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
import pickle
from sklearn.model_selection import GridSearchCV
from mlxtend.preprocessing import minmax_scaling
from sklearn.metrics import precision_score, recall_score

def shuffle(a, b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]

X_train, X_val = [], []
Y_train, Y_val = [], []

noise = True

if(not noise):
	mfcc_features_train = './saved_features_mfcc/train'
	mfcc_features_val = './saved_features_mfcc/val'

	spec_features_train = './saved_features_spectrogram/train'
	spec_features_val = './saved_features_spectrogram/val'
elif(noise):
	mfcc_features_train = './saved_features_mfcc_noise/train'
	mfcc_features_val = './saved_features_mfcc_noise/val'

	spec_features_train = './saved_features_spectrogram_noise/train'
	spec_features_val = './saved_features_spectrogram_noise/val'

feature_type = 'spec'

if(feature_type=='spec'):
	features_train = spec_features_train
	features_val = spec_features_val
elif(feature_type=='mfcc'):
	features_train = mfcc_features_train
	features_val = mfcc_features_val

train_classes = os.listdir(features_train)

for p_label in train_classes:
	files = os.listdir(os.path.join(features_train, p_label))
	for f in files:
		feature = np.load(os.path.join(features_train, p_label, f))
		feature = feature.reshape(-1)
		minmax_scaling(feature, columns=[0])
		feature = np.nan_to_num(feature)
		X_train.append(feature)
		Y_train.append(p_label)

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)

val_classes = os.listdir(features_val)

for p_label in val_classes:
	files = os.listdir(os.path.join(features_val, p_label))
	for f in files:
		feature = np.load(os.path.join(features_val, p_label, f)).reshape(-1)
		minmax_scaling(feature, columns=[0])
		feature = np.nan_to_num(feature)
		X_val.append(feature)
		Y_val.append(p_label)

X_val = np.asarray(X_val)
Y_val = np.asarray(Y_val)

print(X_train.shape, X_val.shape)

parameters = {'kernel':('rbf', 'poly'), 'C':[0.1, 0.5, 1, 3]}

# clf = GridSearchCV(svm1, parameters)

clf = SVC(C=0.5, kernel='poly', gamma='auto')
clf.fit(X_train, Y_train)

if(not noise):
	with open(feature_type + '_svm.pkl','wb') as f:
		pickle.dump(clf, f)
elif(noise):
	with open(feature_type + '_noise_svm.pkl','wb') as f:
		pickle.dump(clf, f)

# svm1 = SVC()
#filename = './saved_models/spec_noise_svm.pkl'

#clf = pickle.load(open(filename, 'rb'))

# print(sorted(clf.cv_results_['kernel']), sorted(clf.cv_results_['C']))

Y_predicted_val = clf.predict(X_val)
print('Accuracy Score: ', accuracy_score(Y_val, Y_predicted_val))
print('Precision: ', precision_score(Y_val, Y_predicted_val, average='weighted'))
print('Recall: ', recall_score(Y_val, Y_predicted_val, average='weighted'))
