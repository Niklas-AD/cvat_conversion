"""
Method to convert output from cvat to use with cityscapesscript

Do not use twice on the same images.
"""
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import re
from cityscapesscripts.helpers.labels import labels as cityscapes_label_list
from tqdm import tqdm

cvat_annotation_export_path = './cityscapes/gtFine/train/karlsruhe'

if glob.glob(os.path.join(cvat_annotation_export_path, '*backup.png')): 
    raise Exception('The method is only supposed to run once')

#generate mapping

cvat_to_cityscapes={}
cvat_to_cityscapes_trainId={}
#map unlabeld to unlabeld
cvat_to_cityscapes[0]=0
cvat_to_cityscapes_trainId[0]=255

with open('label_colors.txt') as f:
    lines=f.readlines()
    for i, _ in enumerate(lines):
        s=lines[i]
        cvat_label_name = re.sub('[^a-zA-Z]+', '', s)
        cvat_label_index =i+1 #line number starts at 1
        
        for label in cityscapes_label_list:
            if label.name == cvat_label_name:
                cityscapes_label_index = label.id  
                cityscapes_label_index_trainingId = label.trainId
                
        cvat_to_cityscapes[cvat_label_index] = cityscapes_label_index 
        cvat_to_cityscapes_trainId[cvat_label_index] = cityscapes_label_index_trainingId
        
#define conversion
f = lambda x: cvat_to_cityscapes[x] 
g = lambda x: cvat_to_cityscapes_trainId[x] 

              
# apply mapping to folder
for image_path in tqdm(glob.glob(os.path.join(cvat_annotation_export_path, '*labelIds.png')), desc="Generating Label Images"):
    
    #backup
    basename = os.path.basename(image_path)
    backup_name = os.path.splitext(basename)[0] + '_backup' + os.path.splitext(basename)[1]
    img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    cv2.imwrite(os.path.join(cvat_annotation_export_path, backup_name), img)
    
    #TrainIDLabel
    train_id_path = os.path.splitext(basename)[0].replace("labelIds","labelTrainIds") + os.path.splitext(basename)[1]
    img_converted_trainID = img_converted = np.vectorize(g)(img)
    cv2.imwrite(os.path.join(cvat_annotation_export_path, train_id_path), img_converted_trainID)
    
    #LabelID
    img_converted = np.vectorize(f)(img)
    cv2.imwrite(image_path, img_converted)
    
#instance ids   
#assumed that attribute is crowd is correctly applied in cvat
for image_path in tqdm(glob.glob(os.path.join(cvat_annotation_export_path, '*instanceIds.png')), desc = "Generating Instance Images"):
    print(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    #backup
    basename = os.path.basename(image_path)
    backup_name = os.path.splitext(basename)[0] + '_backup' + os.path.splitext(basename)[1]
    img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    cv2.imwrite(os.path.join(cvat_annotation_export_path, backup_name), img)
     
    #generate mapping
    cvat_to_cityscapes_instanceId={}
    unique_instances = np.unique(img)
    
    #handle stuff classes
    for instance_id in unique_instances:
        #stuff classes assumes correct handeling in cvat
        if instance_id < 1000:
            cvat_to_cityscapes_instanceId[instance_id] = cvat_to_cityscapes[instance_id]
    
    #generate instance classes
    instance_categories = np.unique((img/1000).astype(int))
    for instance_categorie in instance_categories[instance_categories != 0]:
        lower_bound = instance_categorie*1000
        upper_bound = (instance_categorie+1)*1000      

        for instance_counter, instance in enumerate(unique_instances[(lower_bound <= unique_instances) & (unique_instances < upper_bound)]):
            cvat_to_cityscapes_instanceId[instance] = cvat_to_cityscapes[instance_categorie]*1000+instance_counter
    
    
    print(cvat_to_cityscapes_instanceId)
    
    h = lambda x: cvat_to_cityscapes_instanceId[x] 
    
    img_converted_instanceId  = np.vectorize(h)(img)
    cv2.imwrite(image_path, img_converted_instanceId.astype(np.uint16))


                
