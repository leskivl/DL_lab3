import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.models import load_model
import numpy as np
from PIL import Image

# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()

    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

# define cnn model
def define_model():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def show_imgs(X, labels):
    pyplot.figure(1)
    k = 0
    for i in range(0,4):
        for j in range(0,4):
            val = labels[k]
            pyplot.subplot2grid((8, 4),(i*2,j))
            pyplot.imshow(X[k])
            title = pyplot.title(val[0])
            if val[1] == True:
                pyplot.setp(title, color='green')
            else:
                pyplot.setp(title, color='r')
            pyplot.margins()
            k = k+1
    # show the plot
    pyplot.show()

# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.show()
    # plot accuracy
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.show()

def test(model, testX, testY):
    labels =  ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    indices = np.argmax(model.predict(testX[:16]),1)
    true = np.argmax(testY[:16],1)
    predictions = []
    for i in range(0, len(indices)):
        val = []
        val.append(labels[indices[i]])
        if indices[i] == true[i]:
            val.append(True)
        else:
            val.append(False)
        predictions.append(val)
    show_imgs(testX[:16], predictions)

def load_saved_model():
    # load json and create model
    json_file = open('model_num.json', 'r')

    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("model_num.h5")
    return loaded_model

def save_model(model):
    #save model
    model_json = model.to_json()
    with open("model_num.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model_num.h5")

def train():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # define model
    print(testY[:16])
    model = define_model()
    # fit model
    print("Load model ?  yes/no\n")
    if input() == "yes":
        model  = load_saved_model()
    else:
        epochsCount = 3
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(trainX, trainY, epochs=epochsCount, validation_data=(testX, testY))
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        save_model(model)
        pyplot.plot(np.arange(0, epochsCount), history.history['loss'])
        pyplot.title('Model loss')
        pyplot.ylabel('Loss')
        pyplot.xlabel('Epoch')
        pyplot.legend(['Train'], loc='upper left')
        pyplot.show()
    test(model, testX, testY)

train()





