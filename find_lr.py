# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 21:39:54 2021

@author: Chunayi
"""
from keras.callbacks import ModelCheckpoint,EarlyStopping
from model import *
from data import *
from sklearn.metrics import classification_report, confusion_matrix
from keras_lr_finder import LRFinder
import numpy as np
import cv2

#影像尺寸
input_image_size=256
def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (input_image_size,input_image_size),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator) #组合成一个生成器
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

def validationGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (input_image_size,input_image_size),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    val_image_datagen = ImageDataGenerator(**aug_dict)
    val_mask_datagen = ImageDataGenerator(**aug_dict)
    val_image_generator = val_image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    val_mask_generator = val_mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    val_generator = zip(val_image_generator, val_mask_generator) #组合成一个生成器
    for (val_img,val_mask) in val_generator:
        val_img,val_mask = adjustData(val_img,val_mask,flag_multi_class,num_class)
        yield (val_img,val_mask)


# 定义数据增强的字典
train_data_gen_args = dict(
                    rotation_range=60,
                    fill_mode='wrap')
#產生輸入影像數量為
train_data = trainGenerator(22,'data/membrane/train','image','label',train_data_gen_args,save_to_dir = None)
val_data_gen_args= dict()
validation_data = validationGenerator(5,'data/membrane/validate','image','label',val_data_gen_args,save_to_dir =None)
X_train, y_train = next(train_data)
X_test, y_test = next(validation_data)
val_data_gen_args= dict()


# 载入模型
model = unet(pretrained_weights = None,input_size = (input_image_size,input_image_size,1))
# Adam优化器 binary_crossentropy:交叉熵损失函数
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
lr_finder = LRFinder(model)
# Train a model with batch size 1 for 5 epochs
# with learning rate growing exponentially from 0.0001 to 1
lr_finder.find(X_train, y_train, start_lr=0.00001, end_lr=0.1, batch_size=8, epochs=20)
# Plot the loss, ignore 20 batches in the beginning and 5 in the end
lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
lr_finder.plot_loss_change(sma=20, n_skip_beginning=20, n_skip_end=5, y_lim=(-0.01, 0.01))