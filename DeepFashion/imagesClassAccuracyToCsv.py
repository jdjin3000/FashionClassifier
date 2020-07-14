import csv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = load_model('models/model.h5')
model.summary()

test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_iterator = test_datagen.flow_from_directory('bbox/test', class_mode='categorical', batch_size=32)

filename = test_iterator.filenames[:]

pred = model.predict_generator(test_iterator, verbose=1)

f = open('pred.csv','w', newline='')
wr = csv.writer(f)

wr.writerow(['filename'] + list(test_iterator.class_indices.keys()))

for _filename, acc in zip(filename, pred):
	wr.writerow([_filename] + list(acc))

f.close()