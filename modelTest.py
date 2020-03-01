import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard

anno_path = './Anno'
eval_path = './Eval'
classify_path ='./classify'
bbox_path = './bbox'

model = tf.keras.models.load_model('./models/2ndTest/model.h5')
model.summary()

test_datagen = ImageDataGenerator()
test_iterator = test_datagen.flow_from_directory(os.path.join(bbox_path, "test"), class_mode='categorical', batch_size=16)
tensorboard = TensorBoard(log_dir='./logs/2ndTest')
scores = model.evaluate(test_iterator, callbacks=[tensorboard])

print(scores)