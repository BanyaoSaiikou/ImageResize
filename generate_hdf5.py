#! /usr/bin/env python
# coding: utf-8

import sys
sys.path.append('../')

from pyrr import Quaternion, Matrix44, Vector3, euler
from py_rmpe_server.py_rmpe_config import RmpeGlobalConfig as cf
import numpy as np
import cv2
import os
import os.path
import struct
import h5py
import json
import glob

dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           "..", "cnntest"))
                                           #os.path.abspath(__file__) 作用： 获取当前脚本的完整路径

boxs = []
# box_type = cf.box_types[cf.box_type_idx][0]
box_type = cf.box_types[2][1]

print(box_type)  #green/BoxGreen

tr_hdf5_path = os.path.join(dataset_dir,#os.path.join :	把目录和文件名合成一个路径
                            "ur_train_dataset_" + box_type + ".h5")
val_hdf5_path = os.path.join(dataset_dir, "ur_val_dataset_" + box_type + ".h5")

output_img_dir = os.path.join(dataset_dir, "dataset_output")
os.makedirs(output_img_dir, exist_ok=True)
#os.makedirs() 方法用于递归创建目录。如果子目录创建失败或者已经存在，会抛出一个 OSError 的异常
output_img_dir = os.path.join(output_img_dir, box_type)
os.makedirs(output_img_dir, exist_ok=True)
#os.makedirs() 方法用于递归创建目录。如果子目录创建失败或者已经存在，会抛出一个 OSError 的异常

def process_child(file_list, is_validation):
    file_idx = 0
    for file_dir in file_list:#file_dir循环每个file_list（train_list/test_list）
        img_dir = file_dir + ".png"
        json_dir = file_dir + ".json"
        with open(json_dir, "r") as f:
            img = cv2.imread(img_dir)
            h, w = img.shape[:2]               #获得img的h/w

            data_store = json.load(f)    #解码 JSON 对象
            
            my_box = None
            other_boxs = []
            for box in data_store["objects"]:    #让box循环每一个.json
                if box["class"] == cf.box_types[2][1]:    #cf.box_type_idx =  0，1，2为b，y，g
                    my_box = box
                else:                                                        #检查 目标方块 是否存在
                    other_boxs.append(box)

            sides_3d = []
            sides_2d = []
            if my_box:
                my_box_cuboid_3d = np.asarray(my_box["cuboid"])
                sides_3d.append((my_box_cuboid_3d[0] + my_box_cuboid_3d[2]) / 2)
                sides_3d.append((my_box_cuboid_3d[5] + my_box_cuboid_3d[7]) / 2)
                sides_3d.append((my_box_cuboid_3d[0] + my_box_cuboid_3d[7]) / 2)
                sides_3d.append((my_box_cuboid_3d[2] + my_box_cuboid_3d[5]) / 2)
                sides_3d.append((my_box_cuboid_3d[0] + my_box_cuboid_3d[5]) / 2)
                sides_3d.append((my_box_cuboid_3d[3] + my_box_cuboid_3d[6]) / 2)
                
                my_box_cuboid_2d = np.asarray(my_box["projected_cuboid"])
                sides_2d.append((my_box_cuboid_2d[0] + my_box_cuboid_2d[2]) / 2)
                sides_2d.append((my_box_cuboid_2d[5] + my_box_cuboid_2d[7]) / 2)
                sides_2d.append((my_box_cuboid_2d[0] + my_box_cuboid_2d[7]) / 2)
                sides_2d.append((my_box_cuboid_2d[2] + my_box_cuboid_2d[5]) / 2)
                
                min_center = [0, 0, 0]
                min_dist = -1

                my_box_center = np.asarray(my_box["cuboid_centroid"]) #cuboid_centroid：长方体质心
                count = -1
                for other_box in other_boxs:
                    count += 1
                    other_box_center = np.asarray(other_box["cuboid_centroid"])

                    count_2 = 0
                    min_dist_side = -1
                    min_side_idx = 0
                    for side_3d in sides_3d:
                        dist = np.linalg.norm(other_box_center - side_3d)
                        if min_dist_side == -1 or dist < min_dist_side:
                            min_dist_side = dist
                            min_side_idx = count_2
                        count_2 += 1

                    if min_side_idx >= 4:
                        continue
                    dist = np.linalg.norm(my_box_center - other_box_center)
                    if min_dist == -1 or dist < min_dist:
                        min_dist = dist
                        min_center = other_box_center

                if min_dist == -1:
                    min_center = [0, 0, 0]

                count = 0
                min_dist_side = -1
                min_side_idx = 0
                for side_3d in sides_3d:
                    if count >= 4:
                        break
                    dist = np.linalg.norm(min_center - side_3d)
                    if min_dist_side == -1 or dist < min_dist_side:
                        min_dist_side = dist
                        min_side_idx = count
                    count += 1

                dists_to_cam = []
                # - np.asarray(data_store["camera_data"]["location_worldframe"])
                cam_center = [0, 0, 0]
                count = 0
                for side_3d in sides_3d:
                    if count >= 4:
                        break
                    count += 1
                    dists_to_cam.append(np.linalg.norm(cam_center - side_3d))

                if min_dist == -1:
                    min_side_2d = sides_2d[min_side_idx]
                else:
                    if min_side_idx == 0 or min_side_idx == 1:
                        if dists_to_cam[2] < dists_to_cam[3]:
                            min_side_2d = sides_2d[2]
                        else:
                            min_side_2d = sides_2d[3]
                    else:
                        if dists_to_cam[0] < dists_to_cam[1]:
                            min_side_2d = sides_2d[0]
                        else:
                            min_side_2d = sides_2d[1]

                sorted_idxes = np.argsort(dists_to_cam)
                side_2d_0 = sides_2d[sorted_idxes[0]]
                side_2d_1 = sides_2d[sorted_idxes[1]]

                if min_dist_side > 10:
                    min_side_2d = side_2d_0

                #########
                box_dict = dict()
                if is_validation:
                    box_dict["dataset"] = "UR_Val"
                else:
                    box_dict["dataset"] = "UR_Train"
                box_dict["is_validation"] = is_validation

                box_center = my_box["projected_cuboid_centroid"]
                box_dict["objpos"] = box_center    #中心 

                box_dict["joints"] = np.zeros((4, 3))
                projected_cuboid = np.asarray(my_box["projected_cuboid"])    #投影长方体
                # print(projected_cuboid)

                joint_0 = (projected_cuboid[0] + projected_cuboid[5]) / 2

                box_dict["joints"][0, :2] = joint_0
                box_dict["joints"][0, 2] = 1

                box_dict["joints"][1, :2] = min_side_2d
                box_dict["joints"][1, 2] = 1

                box_dict["joints"][2, :2] = side_2d_0
                box_dict["joints"][2, 2] = 1

                box_dict["joints"][3, :2] = side_2d_1
                box_dict["joints"][3, 2] = 1

                box_h = abs(my_box["bounding_box"]["top_left"][0] -
                            my_box["bounding_box"]["bottom_right"][0])

                box_dict["scale_provided"] = box_h / cf.height

                box_dict["img_width"] = w
                box_dict["img_height"] = h
                box_dict["img_dir"] = img_dir
                box_dict["json_dir"] = json_dir
                box_dict["box_class"] = my_box["class"]

                boxs.append(box_dict)

                # img = cv2.circle(img,
                #                  (int(min_side_2d[0]), int(min_side_2d[1])),
                #                  5, (0, 255, 0), 1)
                # img = cv2.circle(img,
                #                  (int(side_2d_0[0]), int(side_2d_0[1])),
                #                  3, (0, 255, 255), 1)
                # img = cv2.circle(img,
                #                  (int(side_2d_1[0]), int(side_2d_1[1])),
                #                  3, (255, 255, 0), 1)
            else:
                print(file_dir)

            cv2.imwrite(os.path.join(output_img_dir, str(file_idx) + "_" +
                                     os.path.basename(file_dir + ".png")), img)
            file_idx += 1


def process():
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               "..", "cnntest",
                                               "dataset_images"))
    dataset_subdirs = []
    for o in os.listdir(dataset_dir):
        if os.path.isdir(os.path.join(dataset_dir, o)):
            dataset_subdirs.append(os.path.join(dataset_dir, o))

    train_list = []
    test_list = []

    for dataset_subdir in dataset_subdirs:
        count = 0
        n_img = len(glob.glob1(dataset_subdir, "*.json")) - 2
        # print("Num images", n_img)
        for subdir, dirs, files in os.walk(dataset_subdir):
            for file in files:
                file_info = os.path.splitext(file)
                file_name = os.path.splitext(file)[0]
                ext = file_info[-1].lower()
                ignores = ["_camera_settings", "_object_settings"]
                if ext == ".json" and file_name not in ignores:
                    count += 1
                    file_dir = os.path.join(subdir, file_name)
                    if count < (int)(0.8 * n_img):
                        train_list.append(file_dir)
                    else:
                        test_list.append(file_dir)
    print("Start", len(train_list))
    process_child(train_list, False)#以绿色为例 如果图中没有绿色方块 则输出图片路径
    print("Continue", len(test_list))
    process_child(test_list, True)
    print("Done")


def writeHDF5():

    tr_h5 = h5py.File(tr_hdf5_path, "w")
    tr_grp = tr_h5.create_group("datum")
    tr_write_count = 0

    val_h5 = h5py.File(val_hdf5_path, "w")
    val_grp = val_h5.create_group("datum")
    val_write_count = 0

    data = boxs
    n_sample = len(data)

    is_validation_array = [data[i]["is_validation"] for i in range(n_sample)]
    tr_total_write_count = is_validation_array.count(0.0)
    val_total_write_count = len(data) - tr_total_write_count

    print("Num samples", n_sample)
    print("Num training samples", tr_total_write_count)
    print("Num validating samples", val_total_write_count)

    random_order = [i for i, el in enumerate(range(len(data)))]
    # np.random.permutation(n_sample).tolist()

    for count in range(n_sample):
        idx = random_order[count]

        img_dir = data[idx]["img_dir"]

        img = cv2.imread(img_dir)

        is_validation = data[idx]["is_validation"]

        height = img.shape[0]
        width = img.shape[1]
        if (width < 64):
            img = cv2.copyMakeBorder(img, 0, 0, 0, 64 - width,
                                     cv2.BORDER_CONSTANT,
                                     value=(128, 128, 128))
            print("Saving padded image!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            cv2.imwrite("padded_img.jpg", img)
            width = 64

        mask_miss = np.zeros((height, width), dtype=np.uint8)

        # No modify on width, because we want to keep information
        meta_data = np.zeros(shape=(height, width, 1), dtype=np.uint8)
        # print type(img), img.shape
        # print type(meta_data), meta_data.shape

        serializable_meta = {}

        clidx = 0  # Current line index
        # Dataset name (string)
        for i in range(len(data[idx]["dataset"])):
            meta_data[clidx][i] = ord(data[idx]["dataset"][i])
        clidx = clidx + 1
        serializable_meta["dataset"] = data[idx]["dataset"]

        # Image height, image width
        height_binary = float2bytes(data[idx]["img_height"])
        for i in range(len(height_binary)):
            meta_data[clidx][i] = height_binary[i]
        width_binary = float2bytes(data[idx]["img_width"])
        for i in range(len(width_binary)):
            meta_data[clidx][4 + i] = width_binary[i]
        clidx = clidx + 1
        serializable_meta["img_height"] = data[idx]["img_height"]
        serializable_meta["img_width"] = data[idx]["img_width"]

        # (a) is_validation(uint8), numOtherPeople (uint8),
        # people_index (uint8), annolist_index (float),
        # writeCount(float), totalWriteCount(float)
        meta_data[clidx][0] = data[idx]["is_validation"]
        if is_validation:
            count_binary = float2bytes(float(val_write_count))
        else:
            count_binary = float2bytes(float(tr_write_count))
        for i in range(len(count_binary)):
            meta_data[clidx][i] = count_binary[i]
        if is_validation:
            total_write_count_binary = float2bytes(float(val_total_write_count))
        else:
            total_write_count_binary = float2bytes(float(tr_total_write_count))
        for i in range(len(total_write_count_binary)):
            meta_data[clidx][4 + i] = total_write_count_binary[i]
        clidx = clidx + 1
        serializable_meta["is_validation"] = data[idx]["is_validation"]
        if is_validation:
            serializable_meta["count"] = val_write_count
            serializable_meta["total_count"] = val_total_write_count
        else:
            serializable_meta["count"] = tr_write_count
            serializable_meta["total_count"] = tr_total_write_count

        # (b) objpos_x (float), objpos_y (float)
        objpos_binary = float2bytes(data[idx]["objpos"])
        for i in range(len(objpos_binary)):
            meta_data[clidx][i] = objpos_binary[i]
        clidx = clidx + 1
        serializable_meta["objpos"] = [data[idx]["objpos"]]

        # (c) scale_provided (float)
        scale_provided_binary = float2bytes(data[idx]["scale_provided"])
        for i in range(len(scale_provided_binary)):
            meta_data[clidx][i] = scale_provided_binary[i]
        clidx = clidx + 1
        serializable_meta["scale_provided"] = [data[idx]["scale_provided"]]

        # (d) joint (float) (3 line)
        joints = np.asarray(data[idx]["joints"]).T.tolist()
        for i in range(len(joints)):
            row_binary = float2bytes(joints[i])
            for j in range(len(row_binary)):
                meta_data[clidx][j] = row_binary[j]
            clidx = clidx + 1
        serializable_meta["joints"] = [data[idx]["joints"].tolist()]

        for i in range(len(data[idx]["box_class"])):
            meta_data[clidx][i] = ord(data[idx]["box_class"][i])
        clidx = clidx + 1
        serializable_meta["box_class"] = data[idx]["box_class"]

        serializable_meta["img_dir"] = img_dir
        serializable_meta["json_dir"] = data[idx]["json_dir"]

        img4ch = np.concatenate((img, mask_miss[..., None], meta_data), axis=2)
        img4ch = np.transpose(img4ch, (2, 0, 1))

        if is_validation:
            key = "%07d" % val_write_count
            ds = val_grp.create_dataset(key, data=img4ch, chunks=None)
            ds.attrs["meta"] = json.dumps(serializable_meta)
            val_write_count += 1
        else:
            key = "%07d" % tr_write_count
            ds = tr_grp.create_dataset(key, data=img4ch, chunks=None)
            ds.attrs["meta"] = json.dumps(serializable_meta)
            tr_write_count += 1

        # print("Writing sample %d/%d" % (count, n_sample))


def float2bytes(floats):
    if type(floats) is float:
        floats = [floats]
    if type(floats) is int:
        floats = [float(floats)]

    if type(floats) is list and len(floats) > 0 and type(floats[0]) is list:
        floats = floats[0]

    return struct.pack("%sf" % len(floats), *floats)


if __name__ == "__main__":
    process()
    writeHDF5()
