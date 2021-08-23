# example of tending the Restnet50 model
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pickle




#Opening the data file
# 1907 images in the dataset
pickle_in=open("Flavia_x.pickle","rb")
x=pickle.load(pickle_in)
pickle_in=open("Flavia_y.pickle","rb")
y=pickle.load(pickle_in)


num_classes=32
print(np.size(x),np.size(y))

#normalizing
x=x/255.0

#spliting the data into test/training set
trainig_data_percentage=84 # for tuning
test_split_index=int(trainig_data_percentage * len(x) / 100)

x_train=np.array(x[:test_split_index])
y_train=np.array(y[:test_split_index])

x_test=np.array(x[test_split_index:])
y_test=np.array(y[test_split_index:])
print("the number of test samples is:",np.size(y_test))


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)




# load model without classifier layers
model = ResNet50(include_top=False, input_shape=(150, 200, 3))
# mark loaded layers as not trainable
for layer in model.layers:
	layer.trainable = False

# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(64, activation='relu')(flat1)
output = Dense(num_classes, activation='softmax')(class1)
# define new model
model = Model(inputs=model.inputs, outputs=output)
# summarize


model.compile(optimizer = 'adam',
              loss = "categorical_crossentropy",
              metrics=['accuracy'])
model.summary()


#learning
history=model.fit(x_train, y_train, batch_size=100, epochs=10,validation_split=0.187,verbose=1) #validation set in given by: validation_split=percetage from the traing set

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








