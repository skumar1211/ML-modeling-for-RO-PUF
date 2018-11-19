'''code for machine learning models'''
'''created by Sharad Kumar''''

"""
importing libraries
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense,Activation
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.metrics import mean_absolute_error



''' Importing data''''

dataframe = pd.read_csv('data_mypuf.csv')
dataframe1 = pd.read_csv('virginia_tech _challeneg.csv')
dataset = dataframe.values
dataset1 = dataframe1.values

X = dataset[:,0:28]
Y = dataset[:,-1].ravel()

X1 = dataset1[0:255,0:28].astype(int)
Y1= dataset1[0:255,-1].ravel().astype(int)


'''splitting data into training and testing set'''

from sklearn.cross_validation import train_test_split


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.15, random_state = 0)


x_train1,x_test1,y_train1,y_test1 = train_test_split(X1,Y1,test_size = 0.85, random_state = 0)





x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.25, random_state = 0)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.35, random_state = 0)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.45, random_state = 0)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.55, random_state = 0)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.65, random_state = 0)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.75, random_state = 0)

x_train,x_test,y_train,y_test = train_test_split(X1,Y1,test_size = 0.85, random_state = 0)



'''Implementing SVM on the traing set'''

from sklearn.svm import SVC
clf_svc = SVC(kernel = 'linear', random_state = 0)
clf_svc.fit(x_train,y_train)
y_pred_svc = clf_svc.predict(x_test)
cm_svc = confusion_matrix(y_test,y_pred_svc)
fig,ax = plt.subplots()
ax = sn.heatmap(cm_svc, annot=True, fmt="d", cmap="Spectral")

fig.savefig('Confusion_matrix_SVM_Vir_Tech.png')


'''calculating training loss for SVM'''

loss_svm = clf_svc.fit(x_train,y_train).score(x_train,y_train)



''' Implementing Logistic Regression on training set'''

from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression(random_state = 0)

clf_lr.fit(x_train,y_train)

y_pred_lr = clf_lr.predict(x_test)

cm_lr = confusion_matrix(y_test,y_pred_lr)


fig,ax = plt.subplots()

ax = sn.heatmap(cm_lr, annot=True, fmt="d", cmap="Spectral")

fig.savefig('Confusion_matrix_LR_Vir_tech')


'''calculating training loss for LR'''

loss_lr = clf_lr.fit(x_train,y_train).score(x_train,y_train)






''''Implementing Random Forest Classification to the training set''''

from sklearn.ensemble import RandomForestClassifier
clf_rfc = RandomForestClassifier(n_estimators=10,criterion = 'entropy',random_state=0)
a = []

a = clf_rfc.fit(x_train,y_train)
y_pred_rfc = clf_rfc.predict(x_test)
cm_rfc = confusion_matrix(y_test,y_pred_rfc)

fig,ax = plt.subplots()

ax= sn.heatmap(cm_rfc, annot=True, fmt="d", cmap="Spectral")

fig.savefig('Confusion_matrix_RFC')

loss_rfc = clf_rfc.fit(x_train,y_train).score(x_train,y_train)




'''Implementing ANN using hyperparameter Adam'''

clf_ann_ad = Sequential()
clf_ann_ad.add(Dense(32, activation = 'relu', input_dim = 28))
clf_ann_ad.add(Dense(1,activation='relu'))
clf_ann_ad.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
history = clf_ann_ad.fit(X,Y,batch_size = 640, nb_epoch = 1400,validation_split = 0.25)
print(history.history.keys())

fig = plt.figure()

'''PLOTTING ACCURACY CURVE FOR ADAM'''

plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy (Adam)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train','Test'], loc = 'upper left')
fig.savefig('Accuracy_ANN_Adam_Vir_tec.pdf')
plt.show()


'''PLOTTING TRAIN/TEST LOSS FOR ADAM'''

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss (Adam)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train','Test'], loc = 'upper left')
fig.savefig('loss_ANN_Adam_Virtec.png')
plt.show()


answer_train_ad = clf_ann_ad.evaluate(x=x_train,y=y_train)
answer_test_ad = clf_ann_ad.evaluate(x=x_test,y=y_test)

answer_train_ad_V = clf_ann_ad.evaluate(x=x_train1,y=y_train1)
answer_test_ad_V = clf_ann_ad.evaluate(x=x_test1,y=y_test1)


Test_error_ad = answer_test_ad[0]
accuracy_ann_ad = answer_test_ad[1]

train_loss_ad = answer_train_ad[0]

y_pred_ann_ad = clf_ann_ad.predict(x_test)
y_pred_ann_ad = y_pred_ann_ad.round().ravel().astype(int)
cm_ann_ad = confusion_matrix(y_test,y_pred_ann_ad)


Test_error_ad_V = answer_test_ad_V[0]
accuracy_ann_ad_V = answer_test_ad_V[1]

train_loss_ad_V = answer_train_ad_V[0]

y_pred_ann_ad_V = clf_ann_ad.predict(x_test1)
y_pred_ann_ad_V = y_pred_ann_ad_V.round().ravel().astype(int)
cm_ann_ad_V = confusion_matrix(y_test1,y_pred_ann_ad_V)

fig,ax = plt.subplots()
ax = sn.heatmap(cm_ann_ad_V, annot=True, fmt="d", cmap="Spectral")
#fig.savefig('Confusion_matrix_Adam_VirTech_1_act')
fig.savefig('TEST_VIR')




'''Implementing ANN using hyperparameter Adadelta'''

clf_ann_ada = Sequential()
clf_ann_ada.add(Dense(32, activation = 'relu', input_dim = 28))
clf_ann_ada.add(Dense(1,activation='relu'))
clf_ann_ada.compile(optimizer = 'adadelta', loss = 'mean_squared_error', metrics = ['accuracy'])


history = clf_ann_ada.fit(x_train,y_train,batch_size = 640, nb_epoch = 1000, validation_split = 0.25)
print(history.history.keys())

fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy (Adadelta)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train','Test'], loc = 'upper left')
fig.savefig('Accuracy_ANN_Adadelta')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss (Adadelta)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train','Test'], loc = 'upper left')
fig.savefig('loss_ANN_Adadelta')
plt.show()

answer_train_ada = clf_ann_ada.evaluate(x=x_train,y=y_train)
answer_test_ada = clf_ann_ada.evaluate(x=x_test,y=y_test)



Test_error_ada = answer_test_ada[0]
accuracy_ann_ada = answer_test_ada[1]

train_loss_ada = answer_train_ada[0]

y_pred_ann_ada = clf_ann_ada.predict(x_test)
y_pred_ann_ada = y_pred_ann_ada.round().ravel().astype(int)
cm_ann_ada = confusion_matrix(y_test,y_pred_ann_ada)

fig,ax = plt.subplots()
ax = sn.heatmap(cm_ann_ada, annot=True, fmt="d", cmap="Spectral")
fig.savefig('Confusion_matrix_Adadelta')
mean_absolute_error(y_test,y_pred_ann_ada)


'''Implementing ANN using hyperparameter RMSprop'''

clf_ann_rms = Sequential()
clf_ann_rms.add(Dense(32, activation = 'relu', input_dim = 28))
clf_ann_rms.add(Dense(1,activation='relu'))
clf_ann_rms.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['accuracy'])

history = clf_ann_rms.fit(x_train,y_train,batch_size = 640, nb_epoch = 1000,validation_split = 0.25)
print(history.history.keys())

fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy (Rmsprop)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train','Test'], loc = 'upper left')
fig.savefig('Accuracy_ANN_Rmsprop')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss (Rmsprop)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train','Test'], loc = 'upper left')
fig.savefig('loss_ANN_Rmsprop')
plt.show()


answer_train_rms = clf_ann_rms.evaluate(x=x_train,y=y_train)
answer_test_rms = clf_ann_rms.evaluate(x=x_test,y=y_test)

Test_error_rms = answer_test_rms[0]
accuracy_ann_rms = answer_test_rms[1]

train_loss_rms = answer_train_rms[0]

y_pred_ann_rms = clf_ann_rms.predict(x_test)
y_pred_ann_rms = y_pred_ann_rms.round().ravel().astype(int)
cm_ann_rms = confusion_matrix(y_test,y_pred_ann_rms)

fig,ax = plt.subplots()

ax = sn.heatmap(cm_ann_rms, annot=True, fmt="d", cmap="Spectral")
fig.savefig('confusion_matrix_Rmsprop')



'''Implementing ANN using hyperparameter SGD'''

clf_ann_sgd = Sequential()
clf_ann_sgd.add(Dense(32, activation = 'relu', input_dim = 28))
clf_ann_sgd.add(Dense(1,activation='relu'))
clf_ann_sgd.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['accuracy'])
history = clf_ann_sgd.fit(x_train,y_train,batch_size = 640, nb_epoch = 1000,validation_split = 0.25)

print(history.history.keys())

fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy (SGD)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train','Test'], loc = 'upper left')
fig.savefig('Accuracy_ANN_SGD')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss (SGD)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train','Test'], loc = 'upper left')
fig.savefig('loss_ANN_SGD')
plt.show()


answer_train_sgd = clf_ann_sgd.evaluate(x=x_train,y=y_train)
answer_test_sgd = clf_ann_sgd.evaluate(x=x_test,y=y_test)

Test_error_sgd = answer_test_sgd[0]
accuracy_ann_sgd = answer_test_sgd[1]

train_loss_sgd = answer_train_sgd[0]


y_pred_ann_sgd = clf_ann_sgd.predict(x_test)
y_pred_ann_sgd = y_pred_ann_sgd.round().ravel().astype(int)
cm_ann_sgd = confusion_matrix(y_test,y_pred_ann_sgd)

fig,ax = plt.subplots()
ax = sn.heatmap(cm_ann_sgd, annot=True, fmt="d", cmap="Spectral")
fig.savefig('Confusion_matrix_SGD')



         
         
         
         
         
         
         
    
    