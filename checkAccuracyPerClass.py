import os
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator

bbox_path = './bbox'

model = load_model('model.h5')
#model.summary()

test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_iterator = test_datagen.flow_from_directory(os.path.join(bbox_path, "val"), class_mode='categorical', batch_size=32)

print(test_iterator.classes)

#loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print(model.evaluate_generator(test_iterator, steps=2000))