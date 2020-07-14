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

if __name__ == "__main__":
	base_path = 'backgroundRemoval/'

	model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

	for layer in model_resnet.layers:
	    layer.trainable = True

	x = model_resnet.output

	x = Dense(512, activation='elu', kernel_regularizer=l2(0.001))(x)
	y = Dense(14, activation='softmax', name='img')(x)

	final_model = Model(inputs=model_resnet.input, outputs=y)

	print(final_model.summary())

	opt = SGD(lr=0.0001, momentum=0.9, nesterov=True)

	final_model.compile(optimizer=opt,
	                    loss={'img': 'categorical_crossentropy'},
	                    metrics={'img': ['accuracy', TopKCategoricalAccuracy(k=3)]})

	train_datagen = ImageDataGenerator(rotation_range=30.,
	                                   shear_range=0.2,
	                                   zoom_range=0.2,
	                                   width_shift_range=0.2,
	                                   height_shift_range=0.2,
	                                   horizontal_flip=True,
	                                   rescale=1./255,
	                                   validation_split=0.2
	                                   )

	train_iterator = train_datagen.flow_from_directory(base_path, class_mode='categorical',target_size=(224, 224), subset='training')
	test_iterator = train_datagen.flow_from_directory(base_path, class_mode='categorical',target_size=(224, 224), subset='validation')

	lr_reducer = ReduceLROnPlateau(monitor='val_loss',
	                               patience=12,
	                               factor=0.5,
	                               verbose=1)

	#시각화
	tensorboard = TensorBoard(log_dir='./logs')
	#monitored 되는 값의 증가가 멈추면 종료
	early_stopper = EarlyStopping(monitor='val_loss',
	                              patience=30,
	                              verbose=1)
	#가중치 저장. 매 epoch마다 저장됨.
	checkpoint = ModelCheckpoint('./models/model.h5')

	#training
	final_model.fit(train_iterator,
	                          steps_per_epoch=2000,
	                          epochs=200, validation_data=test_iterator,
	                          verbose=2,
	                          shuffle=True,
	                          callbacks=[lr_reducer, checkpoint, early_stopper, tensorboard],
	                          workers=1)


	print("DONE")