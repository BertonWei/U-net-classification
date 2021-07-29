# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 21:39:54 2021

@author: Chunayi
"""
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.models import load_model
from model import *
from data import *
from sklearn.metrics import classification_report, confusion_matrix
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
input_image_size=256
test_file_list=[]
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

# 对测试图片进行规范，使其尺寸和维度上与训练图片保持一致
def testGenerator(test_path,target_size = (input_image_size,input_image_size),flag_multi_class = False,as_gray = True):
    yourPath = test_path
    allFileList = os.listdir(yourPath)
    for file in allFileList:
    #   這邊也可以視情況，做檔案的操作(複製、讀取...等)
    #   使用isdir檢查是否為目錄
    #   使用join的方式把路徑與檔案名稱串起來(等同filePath+fileName)
      if os.path.isdir(os.path.join(yourPath,file)):
        print("I'm a directory: " + file)
    #   使用isfile判斷是否為檔案
      elif os.path.isfile(yourPath+file):
        #紀錄檔案名稱
        test_file_list.append(file)
        img = io.imread(yourPath+file,as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


# 载入模型
model = unet()
# 從 HDF5 檔案中載入模型
model = load_model('./models/Model_1000.001.hdf5')

# 驗證模型
#匯入測試資料集
testGene = testGenerator("./data/membrane/test/")
results = model.predict_generator(testGene,verbose=1)
# 保存測試結果
saveResult(test_file_list,"./data/membrane/test/",results)



