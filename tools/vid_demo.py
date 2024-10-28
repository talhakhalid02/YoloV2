#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger
import cv2
import torch
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.data.datasets.vid_classes import VID_classes
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from val_to_imdb import Predictor
from yolox.models.post_process import post_linking
import random
import json
import REPP

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png", ".JPEG"]

def make_parser():
    parser = argparse.ArgumentParser("YOLOV Demo!")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="/mnt/weka/scratch/yuheng.shi/dataset/VID/ILSVRC2015/Data/VID/snippets/val/ILSVRC2015_val_00130000.mp4", help="path to images or video"
    )

    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default='./exps/yolov++/v++_SwinBaseX_decoupleReg.py',
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default='./V++_outputs/v++_SwinBaseX_decoupleReg/best_ckpt.pth', type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument(
        "--dataset",
        default='vid',
        type=str,
        help="Decide pred classes"
    )
    parser.add_argument("--conf", default=0.05, type=float, help="test conf")
    parser.add_argument("--nms", default=0.5, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=576, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True, 
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--gframe', default=32, help='global frame num')
    parser.add_argument('--lframe', default=0, help='local frame num')
    parser.add_argument('--save_result', default=True)
    parser.add_argument('--post', default=False, action="store_true")
    parser.add_argument('--repp_cfg', default='./tools/yolo_repp_cfg.json', help='repp cfg filename', type=str)
    return parser

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name,[image_name])
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

def imageflow_demo(predictor, vis_folder, current_time, args, exp):
    gframe = exp.gframe_val
    lframe = exp.lframe_val
    traj_linking = exp.traj_linking
    P, Cls = exp.defualt_p, exp.num_classes

    cap = cv2.VideoCapture(args.path)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )

    os.makedirs(save_folder, exist_ok=True)
    ratio = min(predictor.test_size[0] / height, predictor.test_size[1] / width)
    vid_save_path = os.path.join(save_folder, args.path.split("/")[-1])
    img_save_path = save_folder
    logger.info(f"video save_path is {vid_save_path}")
    vid_writer = cv2.VideoWriter(
        vid_save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    frames = []
    outputs = []
    ori_frames = []
    
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            ori_frames.append(frame)
            frame, _ = predictor.preproc(frame, None, exp.test_size)
            frames.append(torch.tensor(frame).to(torch.float16))  # Convert to FP16
        else:
            break

    res = []
    frame_len = len(frames)
    index_list = list(range(frame_len))
    if gframe != 0:
        random.seed(41)
        random.shuffle(index_list)
        random.seed(41)
        random.shuffle(frames)
        split_num = int(frame_len / (gframe))
        for i in range(split_num):
            res.append(frames[i * gframe:(i + 1) * gframe])
        res.append(frames[(i + 1) * gframe:])
    else:
        split_num = int(frame_len / (lframe))
        for i in range(split_num):
            if traj_linking and i != 0:
                res.append(frames[i * lframe - 1:(i + 1) * lframe])
            else:
                res.append(frames[i * lframe:(i + 1) * lframe])
        if traj_linking:
            tail = frames[split_num * lframe - 1:]
        else:
            tail = frames[split_num * lframe:]
        res.append(tail)

    outputs, adj_lists, fc_outputs, names = [], [], [], []
    for ele_id, ele in enumerate(res):
        if ele == []: 
            continue
        frame_num = len(ele)
        ele = torch.stack(ele).to(torch.float16)  # Ensure tensor is in FP16
        t0 = time.time()
        if traj_linking:
            pred_result, adj_list, fc_output = predictor.inference(ele, lframe=frame_num, gframe=0)
            if len(outputs) != 0:  # skip the connection frame
                pred_result = pred_result[1:]
                fc_output = fc_output[1:]
            outputs.extend(pred_result)
            adj_lists.extend(adj_list)
            fc_outputs.append(fc_output)
        else:
            outputs.extend(predictor.inference(ele, lframe=lframe, gframe=gframe))
    if traj_linking:
        outputs = post_linking(fc_outputs, adj_lists, outputs, P, Cls, names, exp)

    outputs = [j for _, j in sorted(zip(index_list, outputs))]
    if args.post:
        logger.info("Post processing...")
        out_post_format = predictor.convert_to_post(outputs, ratio, [height, width])
        out_post = predictor.post(out_post_format)
        outputs = predictor.convert_to_ori(out_post, frame_len)

    logger.info("Saving detection result in {}".format(img_save_path))
    for img_idx, (output, img) in enumerate(zip(outputs, ori_frames[:len(outputs)])):
        if args
