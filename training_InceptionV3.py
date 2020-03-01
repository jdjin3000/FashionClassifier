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

anno_path = './Anno'
eval_path = './Eval'
classify_path ='./classify'
bbox_path = './bbox'


model_InceptionV3 = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
"""
with open('modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model_InceptionV3.summary()
"""

"""
for layer in final_model.layers:
    print(layer.output_shape)
"""

for layer in model_InceptionV3.layers[:-12]:
    layer.trainable =True



x = model_InceptionV3.output
x = Dense(512, activation='elu', kernel_regularizer=l2(0.001))(x)
y = Dense(46, activation='softmax', name='img')(x)

final_model = Model(inputs=model_InceptionV3.input, outputs=y)



#optimizer로 Stochastic Gradient Descent 사용
opt = SGD(lr=0.0001, momentum=0.9, nesterov=True)

#Model.compile에 관하여 : https://keras.io/models/model/
#optimizer : 옵티마이저 선택 
#loss : 손실함수 선택
#img 신경망 출력에서는 categorical_crossentropy를, bbox 출력에는 mse를 사용하겠다는 의미

final_model.compile(optimizer=opt,
                    loss={'img': 'categorical_crossentropy'},
                    metrics={'img': ['accuracy', 'top_k_categorical_accuracy']}) # default: top-5


#데이터에 약간의 변화를 주어 학습할 데이터를 양산하는 것인듯
train_datagen = ImageDataGenerator(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True
                                   )
test_datagen = ImageDataGenerator()

#train_iterator = DirectoryIterator(os.path.join(classify_path, "train"), train_datagen, target_size=(200, 200))
#test_iterator = DirectoryIterator(os.path.join(classify_path, "val"), test_datagen,target_size=(200, 200))

#flow_from_directory에서 알아서 directory 별 클래스를 나눔 여기서 라벨이 붙는것
#train_iterator.class_indices에 번호와 분류가 매핑되어 있고
#train_iterator.classes에 사진별 분류(번호)가 리스트로 저장되어 있다.
train_iterator = train_datagen.flow_from_directory(os.path.join(bbox_path, "train"), class_mode='categorical', batch_size=32)
test_iterator = test_datagen.flow_from_directory(os.path.join(bbox_path, "val"), class_mode='categorical', batch_size=32)

"""
print(dir(train_iterator))
print(train_iterator.directory)
print(train_iterator.class_indices)
print("and len is ...")
print(len(train_iterator.classes))
"""


#콜백함수(ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint)

#ReduceLROnPlateau 설명 : 더 이상 학습으로 개선될 여지가 없을 때 종료	https://keras.io/callbacks/
#monitor : 관찰하고자 하는 항목
#patience : 개선 여지가 없을 때 바로 종료하는 것이 아니라 얼마나 기다려 줄지 (epoch 횟수)
#factor : 학습률 감소 비율 new_lr = lr * factor
#verbose : 얼마나 자세히 정보를 출력할지 0: quiet, 1: update messages.
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
                          steps_per_epoch=1000,
                          epochs=200, validation_data=test_iterator,
                          validation_steps=200,
                          verbose=1,
                          shuffle=True,
                          callbacks=[lr_reducer, checkpoint, early_stopper, tensorboard],
                          #use_multiprocessing=True,
                          workers=1)

#test output

scores = final_model.evaluate_generator(test_iterator, steps=2000)

print(scores)