import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import History
import pickle
import matplotlib.pyplot as plt
import matplotlib


# from vgg import *
# from keras.applications.resnet50 import ResNet50



#Opening the data file
# 1907 images in the dataset
pickle_in=open("Flavia_x_train.pickle","rb")
x_train=pickle.load(pickle_in)
pickle_in=open("Flavia_y_train.pickle","rb")
y_train=pickle.load(pickle_in)
pickle_in=open("Flavia_x_test.pickle","rb")
x_test=pickle.load(pickle_in)
pickle_in=open("Flavia_y_test.pickle","rb")
y_test=pickle.load(pickle_in)

num_classes=32

print("the number of test samples is:",np.size(y_test))
print(y_test,y_train)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



#The mpdel
model=keras.Sequential([
                        keras.layers.Conv2D(64,(4,3),activation='relu',input_shape=x_train.shape[1:]),
                        keras.layers.MaxPooling2D(2,2),
                        keras.layers.Conv2D(64,(4,3),activation='relu'),
                        keras.layers.MaxPooling2D(2,2),
                        keras.layers.Flatten(),
                       keras.layers.Dense(128,activation=tf.nn.relu),
                       keras.layers.Dense(32,activation=tf.nn.softmax)]
                      )

model.compile(optimizer = 'adam',
              loss = "categorical_crossentropy",
              metrics=['accuracy'])
model.summary()

#learning
history=model.fit(x_train, y_train, batch_size=100, epochs=20,validation_split=0.187,verbose=1) #validation set in given by: validation_split=percetage from the traing set

#testing
score=model.evaluate(x_test,y_test,verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
#model.summery()


#-------Ploting Block-------------
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# # save model and architecture to single file
# model.save("model.h5")
# print("Saved model to disk")

# # load model
# model = load_model('model.h5')
# # summarize model.
# model.summary()

# Test loss: 0.8473682403564453
# Test accuracy: 0.7941176295280457