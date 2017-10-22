from __future__ import print_function
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc

import genInput

data  = genInput.GenInput(dir="../cnn/database_128x128")

batchSize = 20
numClasses = len(data.labels)
epochs = 5

inputShape = data.dimensions

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=inputShape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(numClasses, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        #y_pred = self.model.predict_proba(data.inputs["images"]["testing"], verbose=0)
        #rauc = roc_auc_score(data.inputs["labels"]["testing"], y_pred)
        #print("- AUC:", rauc)

history = AccuracyHistory()

model.fit(data.inputs["images"]["training"], data.inputs["labels"]["training"],
          batch_size=batchSize,
          epochs=epochs,
          verbose=1,
          validation_data=(data.inputs["images"]["validation"], data.inputs["labels"]["validation"]),
          callbacks=[history])
score = model.evaluate(data.inputs["images"]["testing"], data.inputs["labels"]["testing"], verbose=0)
y_pred = model.predict_proba(data.inputs["images"]["testing"])
#rauc = roc_auc_score(data.inputs["labels"]["testing"], y_pred)
fp, tp, thr = roc_curve(data.inputs["labels"]["testing"][:,1], y_pred[:,1])
rauc = auc(fp,tp)
print('\nAUC:', rauc)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.plot(range(1, epochs+1), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

plt.title('Receiver Operating Characteristic')
plt.plot(fp, tp, 'b',label='AUC = %0.2f'% rauc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
