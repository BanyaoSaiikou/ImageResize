# -*- coding: utf-8 -*-

import sys
import time
sys.path.append('../')
import os
import re
import math
import keras
import tensorflow.keras.backend as K
import tensorflow as tf



from matplotlib import pyplot as plt
from model import get_training_model
from ds_generators import DataIterator
from optimizers import MultiSGD#!!!!!!!!!!!!!!!1
from py_rmpe_server.py_rmpe_config import RmpeGlobalConfig as cf
#from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger
#from tensorflow.keras.callbacks import TensorBoard, TerminateOnNaN, Callback

#from tensorflow.keras.layers.convolutional import Conv2D
#from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D 

from tensorflow.keras.applications.vgg19 import VGG19
# from keras.backend.tensorflow_backend import set_session
#from IPython.display import clear_output#!!!!1
from glob import glob


#tf.config.experimental_run_functions_eagerly(True)
#tf.compat.v1.disable_eager_execution()


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"#设置log输出信息的，也就是程序运行时系统打印的信息。
#0,1,2,3 四个等级按重要性递增为：
#INFO（通知）<WARNING（警告）<ERROR（错误）<FATAL（致命的）;
#https://blog.csdn.net/c20081052/article/details/90230943


# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

physical_devices =     tf.config.experimental.list_physical_devices('GPU')#physical_devices==GPUs
                                   #我们可以获得当前主机上某种特定运算设备类型（如 GPU 或 CPU ）的列表
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(physical_devices )
#====》》[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

#tf.config.experimental.set_memory_growth(physical_devices[0], True)
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  pass
#设置当前程序可见的设备范围（当前程序只会使用自己可见的设备，不可见的设备不会被当前程序使用）
#此处为[0]
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("###########################")
print("###########################")
print("###########################")
print("###########################")
BATCH_SIZE = 10    #批量大小10 try 30 60
BASE_LR = 2e-6#2e-5
MOMENTUM = 0.9    #势
WEIGHT_DECAY = 5e-4    #權重衰減
LR_POLICY = "step"
GAMMA = 0.333
TRAIN_SAMPLES_L = [14381, 14383, 14383]    #训练样本 L
VAL_SAMPLES_L = [3612, 3612, 3614]    #VAL 样品 L
TRAIN_SAMPLES = TRAIN_SAMPLES_L[cf.box_type_idx] #训练样本
VAL_SAMPLES = VAL_SAMPLES_L[cf.box_type_idx]#VAL 样品
# In original code each epoch is 121746 and step change is on 17th epoch
#在原始代码中，每个纪元是 121746，步长变化是在第 17 个纪元
STEP_SIZE = TRAIN_SAMPLES*17
MAX_ITER = 10
# Set None for 1 gpu, not 1
USE_MULTI_GPUS = None

# WEIGHTS_SAVE = "weights.{epoch:04d}.h5"                                                        #權重存储
# TRAINING_LOG = "training_" + cf.box_types[cf.box_type_idx][0] + ".csv"#训练日志
# LOGS_DIR = "./logs_" + cf.box_types[cf.box_type_idx][0] + "_" + \
#     str(cf.num_parts)                                                                                                       #日志目录
# WEIGHT_DIR = "./weights_" + cf.box_types[cf.box_type_idx][0] + "_" + \
#     str(cf.num_parts)                                                                                                          #權重目录
# LOSS_DIR = "./loss_" + cf.box_types[cf.box_type_idx][0] + "_" + \
#     str(cf.num_parts)                                                                                                             #损失目录

WEIGHTS_SAVE = "weights.{epoch:04d}.h5"                                                        #權重存储
TRAINING_LOG = "training_" + cf.box_types[0][0] + ".csv"#训练日志
LOGS_DIR = "./logs_" + cf.box_types[0][0] + "_" + \
    str(cf.num_parts)                                                                                                       #日志目录
WEIGHT_DIR = "./weights_" + cf.box_types[0][0] + "_" + \
    str(cf.num_parts)                                                                                                          #權重目录
LOSS_DIR = "./loss_" + cf.box_types[0][0] + "_" + \
    str(cf.num_parts)
print("aaaaaaaaaaaaaaaaaa")                                                                                                             #损失目录
print(cf.box_type_idx)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(WEIGHT_DIR, exist_ok=True)
os.makedirs(LOSS_DIR, exist_ok=True)
#exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常。#


def get_last_epoch_and_weights_file():#获取最后一个时期和权重文件
    files = [file for file in glob(WEIGHT_DIR + "/weights.*.h5")]
    files = [file.split("/")[-1] for file in files]
    epochs = [file.split(".")[1] for file in files if file]
    epochs = [int(epoch) for epoch in epochs if epoch.isdigit()]
    if len(epochs) == 0:
        if "weights.best.h5" in files:
            return -1, WEIGHT_DIR + "/weights.best.h5"
    else:
        ep = max([int(epoch) for epoch in epochs])
        return ep, WEIGHT_DIR + "/" + WEIGHTS_SAVE.format(epoch=ep)
    return None, None


model = get_training_model(WEIGHT_DECAY, gpus=USE_MULTI_GPUS)
#keras获得某一层或者某层权重的输出:
from_vgg = dict() # 因为模型定义中的layer的名字与原始vgg名字不同，所以需要调整
from_vgg["conv1_1"] = "block1_conv1"
from_vgg["conv1_2"] = "block1_conv2"
from_vgg["conv2_1"] = "block2_conv1"
from_vgg["conv2_2"] = "block2_conv2"
from_vgg["conv3_1"] = "block3_conv1"
from_vgg["conv3_2"] = "block3_conv2"
from_vgg["conv3_3"] = "block3_conv3"
from_vgg["conv3_4"] = "block3_conv4"
from_vgg["conv4_1"] = "block4_conv1"
from_vgg["conv4_2"] = "block4_conv2"

for layer in model.layers:
    if layer.name in from_vgg:
        layer.trainable = False
# print(model.summary())
# sys.exit()

# Load previous weights or vgg19 if this is the first run
last_epoch, wfile = get_last_epoch_and_weights_file()
if wfile is not None:
    print("Loading %s ..." % wfile)

    model.load_weights(wfile)
    last_epoch = last_epoch + 1

else:
    print("Loading vgg19 weights...")

    vgg_model = VGG19(include_top=False, weights="imagenet")

    for layer in model.layers:
        if layer.name in from_vgg:
            vgg_layer_name = from_vgg[layer.name]
            layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
            print("Loaded VGG19 layer: " + vgg_layer_name)

    last_epoch = 0

# Setup lr multipliers for conv layers
lr_mult = dict()
for layer in model.layers:

    if isinstance(layer, Conv2D):
        # Stage = 1
        if re.match("conv\d_stage1.*", layer.name):
            kernel_name = layer.weights[0].name
            bias_name = layer.weights[1].name
            lr_mult[kernel_name] = 1
            lr_mult[bias_name] = 2

        # Stage > 1
        elif re.match("conv\d_stage.*", layer.name):
            kernel_name = layer.weights[0].name
            bias_name = layer.weights[1].name
            lr_mult[kernel_name] = 4
            lr_mult[bias_name] = 8

        # Vgg
        else:
            kernel_name = layer.weights[0].name
            bias_name = layer.weights[1].name
            lr_mult[kernel_name] = 1
            lr_mult[bias_name] = 2


# Configure loss functions
# Euclidean loss as implemented in caffe
# https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
def eucl_loss(x, y):
    loss = K.sum(K.square(x - y)) / BATCH_SIZE / 2#?????
    return loss


# prepare generators
train_client = DataIterator("../cnntest/ur_train_dataset_" +
                            cf.box_types[0][0] + ".h5",
                            shuffle=True, augment=True, batch_size=BATCH_SIZE)
val_client = DataIterator("../cnntest/ur_val_dataset_" +
                          cf.box_types[0][0] + ".h5",
                          shuffle=True, augment=False, batch_size=BATCH_SIZE)


train_di = train_client.gen()
val_di = val_client.gen()

# Learning rate schedule - equivalent of caffe LR_POLICY = "step"
iterations_per_epoch = TRAIN_SAMPLES // BATCH_SIZE


def step_decay(epoch):
    steps = epoch * iterations_per_epoch * BATCH_SIZE
    lrate = BASE_LR * math.pow(GAMMA, math.floor(steps/STEP_SIZE))
    print("Epoch:", epoch, "Learning rate:", lrate)
    return lrate


print("Weight decay policy...")
for i in range(1, 100, 5):
    step_decay(i)

def clear_output(wait=False):
        """
        Clear the output of the current cell receiving output.

        Parameters
        ----------
        wait : bool, optional [default: False]
            Wait to clear the output until new output is available to replace it.
        """
        if wait:
            # 如果wait为True，等待0.2秒，然后再清除输出
            time.sleep(0.2)
        clear_output(wait=True)
class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(epoch)
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.savefig(os.path.join(LOSS_DIR, "loss_" + str(epoch) + ".png"))
        plt.show()


# Configure callbacks
plot_losses = PlotLosses()
lrate = LearningRateScheduler(step_decay)
checkpoint = ModelCheckpoint(WEIGHT_DIR + "/" + WEIGHTS_SAVE,
                             monitor="loss", verbose=0, save_best_only=False,
                             save_weights_only=True, mode="min", save_freq=1)#period
csv_logger = CSVLogger(TRAINING_LOG, append=True)
tb = TensorBoard(log_dir=LOGS_DIR, histogram_freq=0,
                 write_graph=True, write_images=False)
tnan = TerminateOnNaN()

callbacks_list = [lrate, checkpoint, csv_logger, tb, tnan, plot_losses]

# Sgd optimizer with lr multipliers
multisgd = MultiSGD(lr=BASE_LR, momentum=MOMENTUM,
                    decay=0.0, nesterov=False, lr_mult=lr_mult)

# Start training
if USE_MULTI_GPUS is not None:
    from keras.utils import multi_gpu_model
    model = multi_gpu_model(model, gpus=USE_MULTI_GPUS)

# model.compile(loss=eucl_loss, optimizer=multisgd)
model.compile(loss=eucl_loss, optimizer="SGD")
#model.compile(loss='mean_squared_error', optimizer='adam' )#try this


#fit_generator/Model.fit
model.fit_generator(train_di,
                    steps_per_epoch=iterations_per_epoch,
                    epochs=MAX_ITER,#迭代次数epochs为MAX_ITER=10
                    callbacks=callbacks_list,
                    validation_data=val_di,
                    validation_steps=VAL_SAMPLES // BATCH_SIZE,
                    use_multiprocessing=False,
                    initial_epoch=last_epoch
                    )

