import tensorflow as tf
import os
import random
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
# number of folds for k-fold cross validation
k = 7

# load file paths
covidPositivePaths = os.listdir('COVID-POSITIVE')
covidNegativePaths = os.listdir('COVID-NEGATIVE')

# Start Functions -------------

# Find classification from file name
def getClass(path):
    if path in covidPositivePaths:
        return 0
    else:
        return 1

# get image byte array from path
def loadImage(path, ts=(256, 256)):
    img = tf.keras.preprocessing.image.load_img(path, target_size=ts)
    return tf.keras.preprocessing.image.img_to_array(img)

# Create tensorflow model
def makeModel(transferLearning = False, modelOptimizer='adam', ts=(256, 256)):
    if transferLearning:
        pretrained_model = InceptionV3(
        input_shape = (*ts, 3),
        include_top = False,
        weights = 'imagenet')

        for layer in pretrained_model.layers:
            layer.trainable = False

        last_layer = pretrained_model.get_layer('mixed7')
        #last_layer = pretrained_model.layers[145]
        last_output = last_layer.output
        x = layers.Flatten()(last_output)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(1)(x)
        output = layers.Activation('sigmoid', dtype='float32')(x)

        model = tf.keras.Model(pretrained_model.input, output)
    else:
        # define Convolutional NN
        model = tf.keras.Sequential([tf.keras.layers.Conv2D(16, (3,3), input_shape = (256, 256, 3), activation='relu'),
                                 tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                                 tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
                                 tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                                 tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                 tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                                 tf.keras.layers.Dropout(0.3),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(256, activation = 'relu'),
                                 tf.keras.layers.Dense(128, activation='relu'),
                                 tf.keras.layers.Dropout(.2),
                                 tf.keras.layers.Dense(1, activation = 'sigmoid')])
        # compile model
    model.compile(optimizer = modelOptimizer, loss='binary_crossentropy', metrics=['acc'])
    return model

# End Functions --------------

# merge and shuffle paths
mergedPaths = covidNegativePaths + covidPositivePaths
random.shuffle(mergedPaths)


# split mergedPaths in folds
base_ammount = len(mergedPaths) // k
folds = []
for foldIndex in range(k):
    if foldIndex != (k-1):
        folds.append(mergedPaths[foldIndex*base_ammount:(foldIndex+1)*base_ammount + 1])
    else:
        folds.append(mergedPaths[foldIndex*base_ammount:])

# create k models using k-1 folds for training and k folds for validation
models = []
foldLabels = []
for fold in folds:
    models.append(makeModel(transferLearning = True))
    foldLabel = []
    for item in fold:
        foldLabel.append(getClass(item))
    foldLabels.append(foldLabel)

# train all of the models with their respective folds
foldHistory = []
for i in range(k):
    trainingData, trainingLabels, valData, valLabels = [], [], [], []
    for j in range(k):
        if j!= i:
            trainingData += list(map(lambda x: loadImage('MERGED_FILES/' + x), folds[j]))
            trainingLabels += foldLabels[j]
        else:
            valData += list(map(lambda x: loadImage('MERGED_FILES/' + x), folds[j]))
            valLabels += foldLabels[j]
    print('Training Fold {curr}/{total}'.format(curr = i+1, total=k))

    trainingData = np.array(trainingData)
    trainingLabels = np.array(trainingLabels)
    valData = np.array(valData)
    valLabels = np.array(valLabels)

    history = models[i].fit(trainingData, trainingLabels, epochs=15, validation_data=(valData, valLabels))
    foldHistory.append(history)

for i in range(k):
    models[i].save('exportedModels/modelFold{}.h5'.format(i))
