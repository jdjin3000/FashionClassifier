from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import cv2
import os

if __name__ == "__main__":
	base_path = 'backgroundRemoval/'

	cnn4 = Sequential()
	cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(200,200,3,)))
	cnn4.add(BatchNormalization())

	cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
	cnn4.add(BatchNormalization())
	cnn4.add(MaxPooling2D(pool_size=(2, 2)))
	cnn4.add(Dropout(0.25))

	cnn4.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	cnn4.add(BatchNormalization())
	cnn4.add(Dropout(0.25))

	cnn4.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
	cnn4.add(BatchNormalization())
	cnn4.add(MaxPooling2D(pool_size=(2, 2)))
	cnn4.add(Dropout(0.25))

	cnn4.add(Flatten())

	cnn4.add(Dense(512, activation='relu'))
	cnn4.add(BatchNormalization())
	cnn4.add(Dropout(0.5))

	cnn4.add(Dense(128, activation='relu'))
	cnn4.add(BatchNormalization())
	cnn4.add(Dropout(0.5))

	cnn4.add(Dense(14, activation='softmax'))

	cnn4.compile(optimizer='adam',
	                    loss='categorical_crossentropy',
	                    metrics=['accuracy', TopKCategoricalAccuracy(k=3)])

	train_datagen = ImageDataGenerator(rotation_range=30.,
	                                   shear_range=0.2,
	                                   zoom_range=0.2,
	                                   width_shift_range=0.2,
	                                   height_shift_range=0.2,
	                                   horizontal_flip=True,
	                                   rescale=1./255
	                                   )
	test_datagen = ImageDataGenerator(rescale=1./255)

	train_iterator = train_datagen.flow_from_directory(base_path+'train/', class_mode='categorical',target_size=(200, 200), batch_size=16)
	test_iterator = test_datagen.flow_from_directory(base_path+'validation/', class_mode='categorical',target_size=(200, 200), batch_size=16)
	
	checkpoint = ModelCheckpoint('./models/model.h5')
	tensorboard = TensorBoard(log_dir='./logs')
	early_stopper = EarlyStopping(monitor='val_loss',
                              patience=30,
                              verbose=1)

	#training
	cnn4.fit_generator(train_iterator,
	                          #steps_per_epoch=2000,
	                          epochs=30, validation_data=test_iterator,
	                          #validation_steps=200,
				  verbose=1,
	                          shuffle=True,
	                          callbacks=[checkpoint, tensorboard, early_stopper],
	                          workers=1)


	print("DONE")

