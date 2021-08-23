# example of tending the vgg16 model


from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pickle




num_classes=32

#Opening the data file
# 1907 images in the dataset
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


print(np.shape(x_train),np.shape(y_train))
print(np.shape(x_test),np.shape(y_test))

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


shape=np.shape(x_train)[1:]

# load model without classifier layers
model = VGG16( input_shape=shape,include_top=False)

#avrege layer instead of a flatten layer_ you should remove the ftatten layer in this case
#model = VGG16(include_top=False, input_shape=(150, 200, 3), pooling='avg')


# # mark loaded layers as not trainable
# for layer in model.layers:
# 	layer.trainable = False

# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(128, activation='relu')(flat1)
class2=Dense(64,activation='relu')(class1)
output = Dense(num_classes, activation='softmax')(class2)
# # define new model
model = Model(inputs=model.inputs, outputs=output)

model.compile(optimizer = 'adam',
              loss = "categorical_crossentropy",
              metrics=['accuracy'])
model.summary()


#learning
history=model.fit(x_train, y_train, batch_size=100, epochs=40,validation_split=0.187,verbose=1) #validation set in given by: validation_split=percetage from the traing set

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








