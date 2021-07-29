# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 21:39:54 2021

@author: Chunayi
"""
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras_lr_finder import LRFinder
from model import *
from data import *
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from keras import backend as K
#batch_size
train_data_batch_size=22
test_data_batch_size=5
#影像尺寸
input_image_size=256
#學習率
train_epochs=10
learning_rate=1e-3
#產生影像的倍數
Generator_number=8
#記錄檔案名稱
test_file_list = []
ModelCheckpoint_name="Model_"+str(train_epochs)+str(learning_rate)+".hdf5"

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

#以下為主程式
# 訓練定義數據增強
train_data_gen_args = dict(
                    rotation_range=60,
                    horizontal_flip=True,
                    fill_mode='wrap')
train_data = trainGenerator(train_data_batch_size,'data/membrane/train','image','label',train_data_gen_args,save_to_dir = None)
X_train, y_train = next(train_data)
# 訓練定義數據增強
val_data_gen_args= dict()
validation_data = validationGenerator(test_data_batch_size,'data/membrane/validate','image','label',val_data_gen_args,save_to_dir =None)
X_test, y_test = next(validation_data)
# 载入模型 U-Net
model = unet(pretrained_weights = None,input_size = (input_image_size,input_image_size,1))

# Adam优化器 binary_crossentropy:交叉熵损失函数
model.compile(optimizer = Adam(lr = learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy'])

# 1：保存模型的路径；2：monitor='loss'(检测loss，使其最小)；3：save_best_only=True(只保存在验证集上性能最优的模型)
model_checkpoint = ModelCheckpoint('./models/'+ModelCheckpoint_name, monitor='loss',verbose=1, save_best_only=True)

#學習率遞減
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,verbose=1, mode='min', min_lr=0.0000001)

#避免過擬合方法
#earlyStopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='min')


# 定義訓練回覆學習率的方法
class printlearningrate(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr)
        Epoch_count = epoch + 1
        print('\n', "Epoch:", Epoch_count, ', LR: {:.7f}'.format(lr))
printlr = printlearningrate() 


Unet=model.fit_generator(train_data,
               steps_per_epoch=(len(X_train)/train_data_batch_size)*Generator_number,
               epochs=train_epochs,
               validation_data=validation_data,
               validation_steps=(len(X_test)/test_data_batch_size),
               callbacks=[model_checkpoint,printlr])


#顯示訓練與驗證的accuracy
import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.plot([None] + Unet.history['accuracy'], 'o-')
ax.plot([None] + Unet.history['val_accuracy'], 'x-')
ax.legend(['Train acc', 'Validation acc'], loc = 0)
ax.set_title('Training/Validation acc per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('acc')

#顯示訓練與驗證的loss
import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.plot([None] + Unet.history['loss'], 'o-')
ax.plot([None] + Unet.history['val_loss'], 'x-')
ax.legend(['Train loss', 'Validation loss'], loc = 0)
ax.set_title('Training/Validation loss per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('loss')

#匯入測試資料集
testGene = testGenerator("./data/membrane/test/")
results = model.predict_generator(testGene,verbose=1)
# 保存測試結果
saveResult(test_file_list,"./data/membrane/test/",results)




