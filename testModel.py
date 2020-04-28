from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
model = load_model('models/model.h5')
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_iterator = test_datagen.flow_from_directory('./testModel', class_mode='categorical', batch_size=32)

print(model.evaluate_generator(test_iterator, verbose=1))