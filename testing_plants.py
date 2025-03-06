import plant_training as tp
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import keras as ks 
import json
import sklearn as sk
from sklearn.metrics import confusion_matrix , classification_report
class_name = tp.validation_set.class_names
test_set = tf.keras.utils.image_dataset_from_directory(
    'test',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=1,
    image_size=(128, 128),
    shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
)
y_pred = tp.cnn.predict(test_set) 
predicted_categories = tf.argmax(y_pred, axis=1)

true_categories = tf.concat([y for x, y in test_set], axis=0)
Y_true = tf.argmax(true_categories, axis=1)

cm = confusion_matrix(Y_true,predicted_categories) 