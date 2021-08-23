import numpy as np
import pickle
import cv2
import os


from sklearn.preprocessing import StandardScaler

def get_class_num(img):
    img=img[:-4]
    img=int(img)
    limites=[[1001,1059],
            [1060,1122],
            [1552,1616],
            [1123,1194],
            [1195,1267],
            [1268,1323],
            [1324,1385],
            [1386,1437],
            [1497,1551],
            [1438,1496],
            [2001,2050],
            [2051,2113],
            [2114,2165],
            [2166,2230],
            [2231,2290],
            [2291,2346],
            [2347,2423],
            [2424,2485],
            [2486,2546],
            [2547,2612],
            [2616,2675],
            [3001,3055],
            [3056,3110],
            [3111,3175],
            [3176,3229],
            [3230,3281],
            [3282,3334],
            [3335,3389],
            [3390,3446],
            [3447,3510],
            [3511,3563],
            [3566,3621]]
    for limite in limites:
        if img>=limite[0] and img<=limite[1]:
            return limites.index(limite)

def test_triain_data(x ,y ,number_of_class=32 ,number_of_testing_data_per_class=10):
    shape=list(np.shape(x)[0:])
    shape[0]=number_of_class*number_of_testing_data_per_class
    print('shape:',shape)

    test_x=np.empty(shape=shape)
    test_y=[]

    x_train=np.array(x)

    y_train=list(y)
    itr=0
    for class_num in range(number_of_class):
        idx=-1
        for i in range(number_of_testing_data_per_class):
            idx=y.index(class_num,idx+1)
            x_train = np.delete(x_train, idx,axis=0)
            test_x[itr]=x[idx]
            i+=1
            del y_train[idx]
            test_y.append(y[idx])
    return (x_train,y_train),(test_x,test_y)


#resizing
resize_fact=4*2
image_legth=int(1600/resize_fact)
image_width=int(1200/resize_fact)



# image_width=64
# image_legth=64



path="E:\AI_codes\Leaves"

training_data=[]


def create_training_data():
    for img in os.listdir(path):
        try:
            img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR)
            new_array=cv2.resize(img_array,(image_width,image_legth))
            class_num=get_class_num(img)
            training_data.append([new_array,class_num])
            # plt.imshow(new_array)  # graph it
            # plt.show()
            #break
        except Exception as e:
            pass

create_training_data()

#shuffling
np.random.seed(12321)  # for reproducibility
np.random.shuffle(training_data)

x=[]
y=[]
for features, label in training_data:
    x.append(features)
    y.append(label)

x=np.array(x).reshape(-1,image_width,image_legth,3)

#normalizing
x=x/255.0


# #standarizing
# # created scaler
# scaler = StandardScaler()
# # fit scaler on training dataset
# scaler.fit(trainy)
# # transform training dataset
# trainy = scaler.transform(trainy)
# # transform test dataset
# testy = scaler.transform(testy)




(x_train,y_train),(x_test,y_test)=test_triain_data(x,y)

pickle_out=open("Flavia_x_train.pickle","wb")
pickle.dump(x_train,pickle_out)
pickle_out.close()

pickle_out=open("Flavia_y_train.pickle","wb")
pickle.dump(y_train,pickle_out)
pickle_out.close()

pickle_out=open("Flavia_x_test.pickle","wb")
pickle.dump(x_test,pickle_out)
pickle_out.close()

pickle_out=open("Flavia_y_test.pickle","wb")
pickle.dump(y_test,pickle_out)
pickle_out.close()