from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras import backend as K
import cv2
import os

if __name__ == "__main__":
	base_path = 'C:\\Users\\Jin\\Desktop\\DeepFashion_Data_Test\\img'
	anno_path = 'C:\\Users\\Jin\\Desktop\\DeepFashion_Data_Test\\anno'
	classify_path ='C:\\Users\\Jin\\Desktop\\DeepFashion_Data_Test\\classify'

	#weight='imagenet' : imageNet에서 사전 교육받은 ResNet 모델을 불러옴
	#include_top = False : 네트워크 상단에 완전 연결 신경망을 포함할지 여부
	#pooling: include_top이 false일때만 선택 가능한 옵션 avg 선택시 최종 출력이 2d 텐서
	model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

	#model 객체의 parameter인 layers는 모델을 구성하는 신경망 계층의 평탄화된 리스트(flattened list)이다.
	#[:-12] : 뒤 12 계층 제외한 전부
	for layer in model_resnet.layers[:-12]:
	    layer.trainable = False

	#model.output : 출력 텐서 목록 pooling = 'avg' 였으므로 2d 텐서일 것임
	x = model_resnet.output

	#unit(첫 parameter) : 출력 공간의 차원
	#activation : 활성화 함수 선택. elu 함수 관련 설명 https://reniew.github.io/12/
	#kernel_regularizer : 정규화 유형 선택.과적합을 막기 위함. L2 정규화 선택함
	#정규화 관련 문서 참고 : https://www.tensorflow.org/tutorials/keras/overfit_and_underfit?hl=ko#%EA%B3%BC%EB%8C%80%EC%A0%81%ED%95%A9%EC%9D%84_%EB%B0%A9%EC%A7%80%ED%95%98%EA%B8%B0_%EC%9C%84%ED%95%9C_%EC%A0%84%EB%9E%B5
	#l2(0.001)는 네트워크의 전체 손실에 층에 있는 가중치 행렬의 모든 값이 0.001 * weight_coefficient_value**2만큼 더해진다는 의미

	#카테고리 분류
	x = Dense(512, activation='elu', kernel_regularizer=l2(0.001))(x)
	y = Dense(46, activation='softmax', name='img')(x)

	final_model = Model(inputs=model_resnet.input, outputs=y)

	print(final_model.summary())


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

	train_iterator = DirectoryIterator(os.path.join(classify_path, "train"), train_datagen, target_size=(200, 200))
	test_iterator = DirectoryIterator(os.path.join(classify_path, "val"), test_datagen,target_size=(200, 200))

	
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
	                          steps_per_epoch=2000,
	                          epochs=200, validation_data=test_iterator,
	                          validation_steps=200,
	                          verbose=2,
	                          shuffle=True,
	                          callbacks=[lr_reducer, checkpoint, early_stopper, tensorboard],
	                          #use_multiprocessing=True,
	                          workers=1)

	#test output
	test_datagen = ImageDataGenerator()

	test_iterator = DirectoryIterator(os.path.join(classify_path, "test"), test_datagen, target_size=(200, 200))
	scores = final_model.evaluate_generator(test_iterator, steps=2000)

	print(scores)