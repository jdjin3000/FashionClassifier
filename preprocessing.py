import shutil
import os
import numpy as np
from six.moves import range

#윈도우여도 / 인식함
base_path = 'C:\\Users\\Jin\\Desktop\\DeepFashion_Data_Test\\img'
anno_path = 'C:\\Users\\Jin\\Desktop\\DeepFashion_Data_Test\\anno'
classify_path ='C:\\Users\\Jin\\Desktop\\DeepFashion_Data_Test\\classify'
def process_folders():
    # Read the relevant annotation file and preprocess it
    # Assumed that the annotation files are under '<project folder>/data/anno' path
    with open(os.path.join(anno_path, 'list_eval_partition.txt'), 'r') as eval_partition_file:
        list_eval_partition = [line.rstrip('\n') for line in eval_partition_file][2:]   # 줄마다 하나의 원소로 리스트에 집어넣음 [2:]의 의미는 txt 파일에서 첫줄은 이미지의 갯수를 , 두번째는 attribute의 이름을 알려주기 때문임
        list_eval_partition = [line.split() for line in list_eval_partition]    # line.split()으로 같은 효과 볼 수 있음

        #v[0]은 이미지 파일의 경로이고 [4:]의 뜻은 경로 문자열에서 앞의 4개 문자(0,1,2,3)를 떼겠다는 것 (img/ 를 뗀다는 것)
        #폴더 마지막에 붙은 단어가 category임. 그래서 split으로 떼는거
        #v[1]은 train, test, val 중 하나
        #1. In evaluation status, "train" represents training image, "val" represents validation image, "test" represents testing image
        #list all 첫번째에서 /를 \\으로 바꿈. 리눅스 환경에서 윈도우 환경에 적합하도록 변환한 것
        list_all = [(v[0][4:].replace('/','\\'), v[0].split('/')[1].split('_')[-1], v[1]) for v in list_eval_partition]   
    # Put each image into the relevant folder in train/test/validation folder
    for element in list_all:
        #train, val, test 폴더가 있는가?
        if not os.path.exists(os.path.join(classify_path, element[2])):
            os.mkdir(os.path.join(classify_path, element[2]))
        #train, val, test 폴더 내부에 의류 폴더(Blouse) 등이 있는가?
        if not os.path.exists(os.path.join(os.path.join(classify_path, element[2]), element[1])):
            os.mkdir(os.path.join(os.path.join(classify_path, element[2]), element[1]))
        # 의류 폴더(Blouse) 내부에 원래 분류 해놨던 폴더마냥 소분류를 해놨는가? (ex. Woven_Open-Back_Blouse, Watercolor_Floral_Print_Blouse)
        if not os.path.exists(os.path.join(os.path.join(os.path.join(os.path.join(classify_path, element[2]), element[1])), element[0].split('\\')[0])):
            os.mkdir(os.path.join(os.path.join(os.path.join(os.path.join(classify_path, element[2]), element[1])), element[0].split('\\')[0]))
        
        shutil.copy(os.path.join(base_path, element[0]), os.path.join(os.path.join(os.path.join(classify_path, element[2]), element[1]), element[0]))

process_folders()