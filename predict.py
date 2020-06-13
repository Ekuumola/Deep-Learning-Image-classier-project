# Predict function uses a trained feed-forward convolutional network to predict the name of a flower image provided as an input. The 
# algortihm outputs the probabilities of the top K name classes of the flower image.
#
# Input:
#      - TOPKCLASSES; Number of classes algortihm provider the probabilty estimate
#      - CATEGORYFILE_PATH;  path to .json file containing the mapping of the image integer label codes into flower names
#      - image_path; path to the image to be classified
#      - savedmode_path; path to the flower image classifier Keras model (HDF5 formatted file) 
# 
# Usage: python predict.py [-h] [--top_k TOPKCLASSES]
#                          [--category_names CATEGORYFILE_PATH]
#                          image_path savedmodel_path 
#


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import argparse, pathlib
from PIL import Image
import json
import matplotlib.pyplot as plt

# Support functions

# Covert numpy image to TF tensor, normalize values and resize to match classifier model input size(224 x 224 x 3). 
def process_image(image):
    # Convert numpy array to TF image
    tf_image = tf.convert_to_tensor(image)

    # Normalize image to float values in [0,1]
    tf_image = tf.cast(tf_image, tf.float32)
    tf_image /= 255

    # Resize the image to match flower classifier model input size (224 x 224 x 3):
    tf_image = tf.image.resize(tf_image, [224, 224])
   
    # Convert tensorflow image back to numpy array and return the image
    return tf.keras.preprocessing.image.img_to_array(tf_image)

# Predict image label probabilities for upto top_k categories and return the label numbers and their probabilities
def predict(image, model, top_k):
  
    # Process image to fit the prediction model (nomralize, resize and add extra dimension to represent the batch size)
    processed_test_image = np.expand_dims(process_image(image), axis=0)

    # Use the model and make prediction of the image classes
    model_prediction = model.predict(processed_test_image).squeeze()

    # Identify top k labels, with highest prediction probability
    top_k_labels = np.argpartition(model_prediction, -top_k)[-top_k:]
    top_k_label_probabilities = model_prediction[top_k_labels] 

    # return the array containg the top k label numbers and the probabilities of respective labels
    return top_k_label_probabilities, top_k_labels



# 0) Intitialize commandline parser
# Argument parser
parser = argparse.ArgumentParser(description='Predcit a flower image category name using a Keras neural network model.')

# Positional arguments
parser.add_argument('image_path', type= pathlib.Path, help='Image path and filename')
parser.add_argument('savedmodel_path', type= pathlib.Path, help='Keras model path and filename')

# Optional arguments
parser.add_argument('--top_k', action='store', dest='topkclasses', default = 1, help='top K classes (default: find the best class)')
parser.add_argument('--category_names', action='store', dest ='categoryfile_path',  type= pathlib.Path, default=None, help='Labels to flower names mapping JSON path and filename' )

# 1) Parse arguments and convert to varibles
try:
    results = parser.parse_args()
    image_path = results.image_path
    savedmodel_path = results.savedmodel_path
    categoryfile_path = results.categoryfile_path
    topk = int(results.topkclasses)
except IOError as msg:
    parser.error(str(msg))
    exit()


# 2) Load image 
try:
    im = Image.open(image_path)
    test_image = np.asarray(im)
    print('Loaded image file: ', image_path)
except:
    print('Could not open image file: ', image_path)
    exit()


# 3) Load Keras model
try:
    model = tf.keras.models.load_model(savedmodel_path, custom_objects={'KerasLayer':hub.KerasLayer})
    print('Loaded Keras model file: ', savedmodel_path)
except:
    print('Could not open Keras model file: ', savedmodel_path)
    exit()

# 4) Load category names, if file is available
if categoryfile_path != None:
    try:
        with open(categoryfile_path) as f:
            class_names = json.load(f)
        print('Loaded category file:', categoryfile_path)
    except:
        print('Could not open category file:', categoryfile_path)
else:
    class_names = {} 
    for i in range(0, 102):
        class_names.update({i:'N/A'})

# 5) Predict image class using the Keras model
probs, classes = predict(test_image, model, topk)

# 6) Print results
print_class_names = [class_names.get(str(key+1)) for key in classes]
print('')
print('Most probable flower class for the image %s (top %d classes):'%(image_path, topk))
for i in range(0,topk):
    print('Image class:',classes[i],'\tclass category name:', print_class_names[i], '\t probability: %0.3f' %probs[i])
