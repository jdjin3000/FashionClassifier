from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras import backend as K
import os
from contextlib import redirect_stdout

model_InceptionV3 = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

for layer in model_InceptionV3.layers[:]:
    layer.trainable =True

x = model_InceptionV3.output
x = Dense(512, activation='elu', kernel_regularizer=l2(0.001))(x)
y = Dense(2, activation='softmax', name='img')(x)

final_model = Model(inputs=model_InceptionV3.input, outputs=y)

final_model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

train_datagen = ImageDataGenerator(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   rescale=1.0/255.0
                                   )
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_iterator = train_datagen.flow_from_directory('./binarytest', class_mode='categorical', batch_size=32)
test_iterator = test_datagen.flow_from_directory('./binarytestValidation', class_mode='categorical', batch_size=32)


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
final_model.fit_generator(train_iterator,
                          steps_per_epoch=10,
                          epochs=100, validation_data=test_iterator,
                          validation_steps=10,
                          verbose=1,
                          shuffle=True,
                          callbacks=[lr_reducer, checkpoint, early_stopper, tensorboard],
                          #use_multiprocessing=True,
                          workers=1)