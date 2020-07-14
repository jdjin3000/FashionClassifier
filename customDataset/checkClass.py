# -*- coding: utf-8 -*-
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras import backend as K
import cv2
import os

base_path = 'backgroundRemoval/'


train_datagen = ImageDataGenerator(rotation_range=30.,
	                                   shear_range=0.2,
	                                   zoom_range=0.2,
	                                   width_shift_range=0.2,
	                                   height_shift_range=0.2,
	                                   horizontal_flip=True,
	                                   rescale=1./255
	                                   )

train_iterator = train_datagen.flow_from_directory(base_path+'train/', class_mode='categorical',target_size=(200, 200), batch_size=16)

print(train_iterator.class_indices)

_dict = {}
for v,i in train_iterator.class_indices.items():
	_dict[i] = v

print(_dict)
