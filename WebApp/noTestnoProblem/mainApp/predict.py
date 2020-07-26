import os
import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import tensorflow as tf

path = 'static/exportedModels/'

class imagePredictor:
    def __init__(self):
        # load models
        self.models = []
        loadPath = os.listdir(path)
        for modelFile in loadPath:
            self.models.append(tf.keras.models.load_model(path+modelFile))

    def predictModel(self, imgData):
        predImg = np.expand_dims(imgData, axis=0)
        preds = []
        for model in self.models:
            preds.append(model.predict(predImg)[0][0])
        globalPred = np.array(preds).mean()
        return preds, globalPred

    # def getHeatmap(self, img, layerIndex=0):
    #     sum = 0
    #     predictions = []
    #     models = self.models
    #     img = np.expand_dims(img, axis=0)
    #     for i in range(8):
    #         pred = models[i].predict(img)[0]
    #         sum += pred
    #         predictions.append(pred)
    #
    #     model = models[predictions.index(max(predictions))] # best model
    #     argmax = 0
    #     print("model index:",argmax)
    #     output = model.output[:, argmax]
    #     #retrieve last convolutional layer
    #     last_conv_layer = model.layers[4]
    #     tf.compat.v1.disable_eager_execution()
    #     #Get Convolutional Network Gradient
    #     grads = K.gradients(output, last_conv_layer.output)[0]
    #
    #     #get the average intensity of each convolution batch region
    #     pooled_grads = K.mean(grads, axis=(0, 1, 2))
    #
    #     #retrive pooled valued
    #     iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    #     pooled_grads_value, conv_layer_output_value = iterate([x])
    #     for i in range(512):
    #         conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    #
    #     #plot heatmap
    #     heatmap = np.mean(conv_layer_output_value, axis=-1)
    #     heatmap = np.maximum(heatmap, 0)
    #     heatmap /= np.max(heatmap)
    #     plt.matshow(heatmap)
    #     plt.show()
    #     img = imgData
    #
    #     #resize heatmat to match image and convert to RGB
    #     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    #     heatmap = np.uint8(255 * heatmap)
    #
    #     #overlay with original image
    #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    #
    #     overlay_intensity = .8
    #     superimposed_img = heatmap * overlay_intensity + img
    #     return superimposed_img
