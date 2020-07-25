import os
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from keras.applications.inception_v3 import preprocess_input
def predict(img, path):
    sum = 0
    predictions = []
    models = os.listdir("models")
    for i in range(10):
        pred = models[i].predict(img)[0]
        sum += pred
        predictions.append(pred)
    preds = round(sum / 10)
    img = np.expand_dims(img, axis=0)
    x = preprocess_input(img)
    model = models[predictions.index(max(predictions))] # best model
    argmax = np.argmax(preds[0])
    print("model index:",argmax)
    output = model.output[:, argmax]
    #retrieve last convolutional layer
    last_conv_layer = model.get_layer('block5_conv3')

    #Get Convolutional Network Gradient
    grads = K.gradients(output, last_conv_layer.output)[0]

    #get the average intensity of each convolution batch region
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    #retrive pooled valued
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    #plot heatmap
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    plt.show()
    img = cv2.imread(path)

    #resize heatmat to match image and convert to RGB
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    #overlay with original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay_intensity = .8
    superimposed_img = heatmap * overlay_intensity + img
    output = '/content/drive/My Drive/output.jpg'
    cv2.imwrite(output, superimposed_img)
    img= mpimg.imread(output)
    plt.imshow(img)
    plt.axis('off')
    plt.title(predictions.loc[0,'category'])
    return predictions + [preds]
