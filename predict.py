# -*- coding: utf-8 -*-

"""
Created on Mon Nov 25 18:08:12 2019

@author: tuan
"""
import sys
from tkinter import Y
import cv2
import time
import numpy as np
#import util
import os
import tensorflow as tf
import config_reader
#from config_reader import config_reader
from py_rmpe_server.py_rmpe_config import RmpeGlobalConfig as cf
from scipy.ndimage.filters import gaussian_filter
from model import get_testing_model
# from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
import math
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.log_device_placement = True
# sess = tf.Session(config=config)
# set_session(sess)

#physical_devices = tf.config.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
physical_devices =     tf.config.experimental.list_physical_devices('GPU')#physical_devices==GPUs
                                   #我们可以获得当前主机上某种特定运算设备类型（如 GPU 或 CPU ）的列表
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(physical_devices )


weight_file_idx = "0050"
colors = [(255, 0, 255), (255, 0, 255), (0, 255, 255)]
# weight_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
#                                           "training",
#                                           "weights_" +
#                                           cf.box_types[cf.box_type_idx][1] +
#                                           "_" + str(cf.num_parts)))
weight_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                          "training",
                                          "weights_" +
                                          cf.box_types[0][1] +
                                          "_" + str(cf.num_parts)))


def process(input_img, model, params, model_params):
    multiplier = [x * model_params["boxsize"] / input_img.shape[0]#box的尺寸/画面宽度的尺寸
                  for x in params["scale_search"]]#x 等于 0或1
    heatmap_avg = np.zeros((input_img.shape[0], input_img.shape[1],
                            cf.num_parts_with_background))#360x640 矩阵 由0组成
    # print("heatmap_avg:",heatmap_avg)
    # print("0:",input_img.shape[0])   0: 360 宽
    # print("1:",input_img.shape[1])   1: 640   高                                                                     num_parts_with_background: 5
    # print("num_parts_with_background:",cf.num_parts_with_background)
    #print("multiplier:",multiplier)                          [1.0222222222222221]
    #print("len(multiplier):",len(multiplier))        len(multiplier):1
    #print(model_params["boxsize"],model_params["boxsize"] )        368
   


    for m in range(len(multiplier)):
        scale = multiplier[m]
        #print("m:",m)     m:0
        #print("scale:",scale)     scale:1.0222222222222221
        img_to_test = cv2.resize(input_img, (0, 0), fx=scale, fy=scale,
                                 interpolation=cv2.INTER_CUBIC)#放缩使用cv.INTER_CUBIC(较慢)
        #print("img to test:",img_to_test)一样大
        #print("input_img",input_img)一样大
        img_padded, pad = util.padRightDownCorner(img_to_test,
                                                  model_params["stride"],#右下角填充灰色，使宽、高像素是8的倍数。
                                                  model_params["padValue"])
        #print("1",img_padded)
        #print("2",pad)     [0, 0, 0, 5]/[0, 0, 0, 2]


        # Required shape (1, width, height, channels)
        trans_in_img = np.transpose(np.float32(img_padded[:, :, :, np.newaxis]),
                                    (3, 0, 1, 2))#T huahua



        output_blobs = model.predict(trans_in_img)# 预测样本属于每个类别的概率；此处用到了训练好的模型0050
        #print(model.predict(trans_in_img))        # 打印概率
        #print(np.argmax(model.predict(trans_in_img), axis=1))        # 打印最大概率对应的标签


        # Extract outputs, resize, and remove padding   #提取输出、调整大小和删除填充
        heatmap = np.squeeze(output_blobs[0])  # Output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params["stride"],
                             fy=model_params["stride"],
                             interpolation=cv2.INTER_CUBIC)#output的宽高都缩小了8倍，这里恢复到与input_img相同。
        heatmap = heatmap[:img_padded.shape[0] - pad[2],
                          :img_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (input_img.shape[1], input_img.shape[0]),
                             interpolation=cv2.INTER_CUBIC)




        # heatmap_avg = heatmap_avg + heatmap
        heatmap_avg = np.maximum(heatmap_avg, heatmap)#对比 取大的；逐位比较取其大者
    
    # for part in range(cf.num_parts):
    #     map_ori = heatmap_avg[:, :, part]
    #     map_2 = gaussian_filter(map_ori, sigma=3)

    # heatmap_avg = heatmap_avg / len(multiplier)

    all_peaks = []
    all_peaks_with_score_and_id = []
    peak_counter = 0

    for part in range(cf.num_parts):#print(cf.num_parts) =4      parts = ["top", "side", "side_0", "side_1"]
        
        map_ori = heatmap_avg[:, :, part]
        
        map = gaussian_filter(map_ori, sigma=3)#高斯去噪

#找到峰值（当前像素值（0-255）大小比上下左右的都大）
        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up,
             map >= map_down, map > params["thre1"]))#这里输出：像素值都是T or F，峰值T，图像大小和原图一样
        peaks = list(zip(np.nonzero(peaks_binary)[1],
                         np.nonzero(peaks_binary)[0]))  # Note reverse
        # print("peaks",peaks)
# Note reverse# 输出T的坐标，即是峰值的一系列坐标 [(h1, w1), (h2, w2), (h3, w3), (h4, w4)]，此处坐标与原图是反转的(x,y反转了)？？

        if(len(peaks) > 0):
            #print("len(peaks)",len(peaks))                   2                                1                                       4
            #print("peaks",peaks)                     [(370, 80), (26, 186)]         [(343, 88)]                peaks [(340, 93), (368, 106), (0, 275), (26, 281)]
            max_score = 0
            max_peaks = peaks[0]
            for x in peaks:
                # print("x",x) 
                # print("^^^^^^^^^^")
                if map_ori[x[1], x[0]] >= max_score:#抽出的几个点 分别对应到mapori上 如果为白色 则作为峰值提取
                    max_score = map_ori[x[1], x[0]]#最初点的颜色不为黑色 就为max_score ；for循环 取得最大点
                    max_peaks = x
                   # print("x;",x[0],x[1])
            #print("max_peaks",max_peaks) 

            peaks = list([max_peaks])
            #print("peaks",peaks)  
        all_peaks.append(peaks)
        #所有part的峰值全存入

        #all_peaks=[ [((h0, w0, s0,0),(h1, w1, s1,1)....]\  第一个part的所有值

        #            [((hi, wi, si,i),(hi+1, wi+1, si+1,i+1)....]\

        #               .....

        # ]
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],)
                                   for i in range(len(id))]

        all_peaks_with_score_and_id.append(peaks_with_score_and_id)
        peak_counter += len(peaks)#个数

    print(all_peaks)#！！！！！！！！！！！！！！！！！！！！1，四个坐标由他输出 调查all_peaks的意思
                                 # 2，print(all_peaks[0])/ print(all_peaks[1])/ print(all_peaks[2])/ print(all_peaks)[3]
                                  #3，[[(354, 136)], [], [(26, 281)], []]  出现空集 分析一下原因
    # cv2.imshow('image',all_peaks[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    a=list()
    b=[]
    c=[]
    d=[]
    L=0
    k=[]
    l=[]
    x1=[]
    y=[]
    theta=[]
    x2=[]
    y2=[]
    k2=[]
    
    for i, peaks in enumerate(all_peaks):
        # print("i:",i)
        # print("peaks:",peaks)

        if(len(peaks) > 0):
            if i == 0:
                input_img = cv2.circle(input_img, peaks[0], 4, colors[0], -1)#画圆 如果有有两个 则输出两个
                #print("picture1:",peaks[0])
                a.append(peaks[0][0])
                b.append(peaks[0][1])
                
               
            elif i == 1:
                input_img = cv2.circle(input_img, peaks[0], 3, colors[0], 2)
                #print("picture2:",peaks[0])
                c.append(peaks[0][0])
                d.append(peaks[0][1])
            #      print("c",c,"d",d)
            # print("a",a,"b",b)
    print("a",a,"b",b)
    print("c",c,"d",d)
    if len(a)>0:
        k=(int(b[0])-int(d[0]))/(int(a[0])-int(c[0]))
        l=b[0]-(a[0]*b[0]-a[0]*d[0])/(a[0]-c[0])
        y=l

        if (k) != 0:
            x1=-1*(l)/(k)
            print("k",k)
            print("l",l)
            print("y",y)
            print("x",x1)
            k2=-1/k
            print("k2",k2)
            x2=l/(k2-k)
            y2=k2*x2
            print("x2",x2)
            print("y2",y2)




            theta=(math.atan(abs(y2)/abs(x2)))*180/math.pi
            print("theta",theta)
        else:
            print("k",k)
            print("l",l)
            print("y",y)
            # print("x",x)
            print("k==0")
            theta=90
            print("theta",theta)

        


            
            # elif i == 2:
            #     input_img = cv2.circle(input_img, peaks[0], 5, colors[0], 2)
            # elif i == 3:
            #     input_img = cv2.circle(input_img, peaks[0], 6, colors[0], 2)
            # else:
            #     input_img = cv2.circle(input_img, peaks[0], 2, colors[1], -1)
        # cv2.imshow('image', input_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    # plt.figure(1)
    # plt.subplot(211)
    # plt.imshow(gaussian_filter(heatmap_avg[:, :, 0], sigma=3))
    # plt.subplot(212)
    # plt.imshow(canvas)
    # plt.show()

    # # gaussian_filter(heatmap_avg[:, :, 0], sigma=3)
    # result = heatmap_avg[:, :, 1]
    # return result / result.max() * 256
    return input_img


def process_from_image():

    input_folder = "real_7"
    output_folder = "real_7_out_shan"
    keras_weights_file = os.path.join(weight_dir,
                                      "weights." + weight_file_idx + ".h5")

    tic = time.time()
    print("Start processing...")

    # load model
    model = get_testing_model()
    model.load_weights(keras_weights_file)

    print("Continue processing...")
    # run
    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             input_folder))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              output_folder))
    os.makedirs(output_dir, exist_ok=True)
    
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            file_info = os.path.splitext(file)
            file_name = os.path.splitext(file)[0]
           
            ext = file_info[-1].lower()
            
            ignores = ["_camera_settings", "_object_settings"]#!!!!!!!!!!!!!!!!!!!! 输出的四个坐标分别是什么 怎么得到的 怎么输出来
            #print("ignores:",ignores)   ignores: ['_camera_settings', '_object_settings']
   
            if (ext == ".json" and file_name not in ignores) or ext == ".jpg":
                ext = ".jpg" if ext == ".jpg" else ".png"
                input_img_dir = os.path.join(input_dir, file_name + ext)
                output_image_dir = os.path.join(output_dir, file_name + ext)
                params, model_params = config_reader()
                input_img = cv2.imread(input_img_dir)
                
                canvas = process(input_img, model, params, model_params)
                
                cv2.imwrite(output_image_dir, canvas)

    toc = time.time()
    print("Processing time is %.5f" % (toc - tic))


# def process_from_video():

#     input_video = "tuantd_rgb.avi"
#     #input_video = "test0001.mp4"
#     output_video = "tuantd_rgb_out_gazebo_shan.avi"
#     keras_weights_file = os.path.join(weight_dir,
#                                       "weights." + weight_file_idx + ".h5")

#     tic = time.time()
#     print("Start processing...")

#     # load model
#     model = get_testing_model()
#     model.load_weights(keras_weights_file)

#     print("Continue processing...")
#     # run
#     cap = cv2.VideoCapture(input_video)
#     if cap.isOpened() is False:
#         print("Error opening video file")

#     frame_width = 640
#    # frame_height = 360
#     # frame_width = 640
#     frame_height = 640
#     out = cv2.VideoWriter(output_video,
#                           cv2.VideoWriter_fourcc("M", "J", "P", "G"),
#                           10, (frame_width, frame_height))

#     count = 0
#     while(cap.isOpened()):
#         # Capture frame by frame
#         ret, frame = cap.read()
#         if ret is True:

#             count += 1
#             if count % 5 != 0:
#                 continue

#             params, model_params = config_reader()
#             frame = cv2.resize(frame, (frame_width, frame_height))
#             if count >= 30:
#                 frame = process(frame, model, params, model_params)
#             out.write(frame)

#             cv2.imshow("Output", frame)

#             # Press Q on keyboard to stop recording
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break
#         else:
#             break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     toc = time.time()
#     print("Processing time is %.5f" % (toc - tic))


# def process_all_objects_child(input_img_0, models):

#     output_img = input_img_0.copy()

#     for model_idx in range(len(models)):
#         model = models[model_idx]

#         input_img = input_img_0.copy()

#         params, model_params = config_reader()

#         multiplier = [x * model_params["boxsize"] / input_img.shape[0]
#                       for x in params["scale_search"]]
#         heatmap_avg = np.zeros((input_img.shape[0], input_img.shape[1],
#                                 cf.num_parts_with_background))

#         for m in range(len(multiplier)):

#             scale = multiplier[m]
#             img_to_test = cv2.resize(input_img, (0, 0), fx=scale, fy=scale,
#                                      interpolation=cv2.INTER_CUBIC)
#             img_padded, pad = util.padRightDownCorner(img_to_test,
#                                                       model_params["stride"],
#                                                       model_params["padValue"])

#             # Required shape (1, width, height, channels)
#             trans_in_img = np.transpose(
#                 np.float32(img_padded[:, :, :, np.newaxis]), (3, 0, 1, 2))

#             output_blobs = model.predict(trans_in_img)

#             # Extract outputs, resize, and remove padding
#             heatmap = np.squeeze(output_blobs[0])  # output 1 is heatmaps
#             heatmap = cv2.resize(heatmap, (0, 0), fx=model_params["stride"],
#                                  fy=model_params["stride"],
#                                  interpolation=cv2.INTER_CUBIC)
#             heatmap = heatmap[:img_padded.shape[0] - pad[2],
#                               :img_padded.shape[1] - pad[3], :]
#             heatmap = cv2.resize(heatmap,
#                                  (input_img.shape[1], input_img.shape[0]),
#                                  interpolation=cv2.INTER_CUBIC)

#             # heatmap_avg = heatmap_avg + heatmap
#             heatmap_avg = np.maximum(heatmap_avg, heatmap)

#         all_peaks = []
#         all_peaks_with_score_and_id = []
#         peak_counter = 0

#         for part in range(cf.num_parts):
#             map_ori = heatmap_avg[:, :, part]
#             map = gaussian_filter(map_ori, sigma=3)

#             map_left = np.zeros(map.shape)
#             map_left[1:, :] = map[:-1, :]
#             map_right = np.zeros(map.shape)
#             map_right[:-1, :] = map[1:, :]
#             map_up = np.zeros(map.shape)
#             map_up[:, 1:] = map[:, :-1]
#             map_down = np.zeros(map.shape)
#             map_down[:, :-1] = map[:, 1:]

#             peaks_binary = np.logical_and.reduce(
#                 (map >= map_left, map >= map_right, map >= map_up,
#                  map >= map_down, map > params["thre1"]))
#             peaks = list(zip(np.nonzero(peaks_binary)[1],
#                              np.nonzero(peaks_binary)[0]))  # Note reverse

#             if(len(peaks) > 0):
#                 max_score = 0
#                 max_peaks = peaks[0]
#                 for x in peaks:
#                     if map_ori[x[1], x[0]] >= max_score:
#                         max_score = map_ori[x[1], x[0]]
#                         max_peaks = x

#                 peaks = list([max_peaks])

#             all_peaks.append(peaks)
#             peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
#             id = range(peak_counter, peak_counter + len(peaks))
#             peaks_with_score_and_id = [peaks_with_score[i] + (id[i],)
#                                        for i in range(len(id))]

#             all_peaks_with_score_and_id.append(peaks_with_score_and_id)
#             peak_counter += len(peaks)

#         print(all_peaks)

#         for i, peaks in enumerate(all_peaks):
#             if(len(peaks) > 0):
#                 if i == 0:
#                     output_img = cv2.circle(output_img, peaks[0], 4,
#                                             colors[0], -1)
#                 elif i == 1:
#                     output_img = cv2.circle(output_img, peaks[0], 3,
#                                             colors[0], 2)
#                 # else:
#                 #     output_img = cv2.circle(output_img, peaks[0], 2,
#                 #                             colors[1], -1)

#     return output_img


# def process_all_objects():

#     input_folder = "real_7"
#     output_folder = "real_7_out_allshan"

#     models = []
#     for i in range(len(cf.box_types)):
#         weight_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
#                                                   "..",
#                                                   "training",
#                                                   "weights_" +
#                                                   cf.box_types[i][1] + "_" +
#                                                   str(cf.num_parts)))
#         keras_weights_file = os.path.join(weight_dir, "weights." +
#                                           weight_file_idx + ".h5")

#         model = get_testing_model()
#         model.load_weights(keras_weights_file)
#         models.append(model)

#     input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
#                                              input_folder))
#     output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
#                                               output_folder))
#     os.makedirs(output_dir, exist_ok=True)
#     for subdir, dirs, files in os.walk(input_dir):
#         for file in files:
#             file_info = os.path.splitext(file)
#             file_name = os.path.splitext(file)[0]
#             ext = file_info[-1].lower()
#             ignores = ["_camera_settings", "_object_settings"]
#             if (ext == ".json" and file_name not in [ignores]) or ext == ".jpg":
#                 ext = ".jpg" if ext == ".jpg" else ".png"
#                 input_img_dir = os.path.join(input_dir, file_name + ext)
#                 output_image_dir = os.path.join(output_dir, file_name + ext)

#                 input_img = cv2.imread(input_img_dir)
#                 canvas = process_all_objects_child(input_img, models)
#                 cv2.imwrite(output_image_dir, canvas)


if __name__ == "__main__":
     process_from_image()
    #process_from_video()
     #process_all_objects()
