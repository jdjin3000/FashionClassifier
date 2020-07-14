import shutil
import os
import numpy as np
import cv2
from six.moves import range

anno_path = './Anno'
eval_path = './Eval'
classify_path ='./classify'
bbox_path = './bboxForTsne'

def seperateImages():
    #Training Set, Validation Set, Testing Set 분리
    with open(os.path.join(eval_path, 'list_eval_partition.txt'), 'r') as eval_file, \
        open(os.path.join(anno_path, 'list_bbox.txt'), 'r') as bbox_file:
        eval_data = [line.rstrip('\n') for line in eval_file][2:]
        eval_list = [line.split() for line in eval_data]

        bbox_data = [line.rstrip('\n') for line in bbox_file][2:]
        bbox_list = [line.split() for line in bbox_data]

        #(이미지 경로, 카테고리, 소속 세트, (Boundary Box 좌표:x_1  y_1  x_2  y_2))
        lists = [(os.path.join('.',eval_[0]), eval_[0].split('/')[1].split('_')[-1], eval_[1], (int(bbox_[1]), int(bbox_[2]), int(bbox_[3]), int(bbox_[4]))) \
        for (eval_, bbox_) in zip(eval_list, bbox_list)]

    for element in lists:
        if not os.path.exists(bbox_path):
            os.mkdir(bbox_path)
        if not os.path.exists(os.path.join(bbox_path, element[2])):
            os.mkdir(os.path.join(bbox_path, element[2]))
        if not os.path.exists(os.path.join(os.path.join(bbox_path, element[2]), element[1])):
            os.mkdir(os.path.join(os.path.join(bbox_path, element[2]), element[1]))
        
        img = cv2.imread(os.path.join('.', element[0]))
        cropped_img = img[element[-1][1]:element[-1][3], element[-1][0]:element[-1][2]]
        cv2.imwrite(os.path.join(os.path.join(os.path.join(bbox_path, element[2]), element[1]), element[0].split('/')[2]), cropped_img)

    print("Cropping Image has done Successfully!")
        

seperateImages()
