# -*- coding: utf-8 -*-
# @Author  : Abner
# @Time    : 2018/12/19

import os
from scipy import misc as scisc
import cv2
import numpy as np
from warnings import warn
from time import sleep
import argparse

from multiprocessing import Pool
from multiprocessing import TimeoutError as MP_TimeoutError

START = "START"
FINISH = "FINISH"
WARNING = "WARNING"
FAIL = "FAIL"


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--input_path', default='', type=str,
                    help='Root path of raw dataset.')
parser.add_argument('--output_path', default='', type=str,
                    help='Root path for output.')
parser.add_argument('--log_file', default='./pretreatment_oumvlp.log', type=str,
                    help='Log file path. Default: ./pretreatment_oumvlp.log')
parser.add_argument('--log', default=False, type=boolean_string,
                    help='If set as True, all logs will be saved. '
                         'Otherwise, only warnings and errors will be saved.'
                         'Default: False')
parser.add_argument('--worker_num', default=32, type=int,
                    help='How many subprocesses to use for data pretreatment. '
                         'Default: 1')
opt = parser.parse_args()

INPUT_PATH = opt.input_path
OUTPUT_PATH = opt.output_path
IF_LOG = opt.log
LOG_PATH = opt.log_file
WORKERS = opt.worker_num
print('WORKERS--',WORKERS)
T_H = 64
T_W = 64


def log2str(pid, comment, logs):
    str_log = ''
    if type(logs) is str:
        logs = [logs]
    for log in logs:
        str_log += "# JOB %d : --%s-- %s\n" % (
            pid, comment, log)
    return str_log


def log_print(pid, comment, logs):
    str_log = log2str(pid, comment, logs)
    if comment in [WARNING, FAIL]:
        with open(LOG_PATH, 'a') as log_f:
            log_f.write(str_log)
    if comment in [START, FINISH]:
        if pid % 500 != 0:
            return
    print(str_log, end='')


def cut_img(img, seq_info, frame_name, pid):
    # A silhouette contains too little white pixels
    # might be not valid for identification.
    if img.sum() <= 10000:
        message = 'seq:%s, frame:%s, no data, %d.' % (
            '-'.join(seq_info), frame_name, img.sum())
        warn(message)
        log_print(pid, WARNING, message)
        return None
    # Get the top and bottom point
    y = img.sum(axis=1)
    y_top = (y != 0).argmax(axis=0)
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]
    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    _r = img.shape[1] / img.shape[0]
    _t_w = int(T_H * _r)
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)
    # Get the median of x axis and regard it as the x center of the person.
    sum_point = img.sum()
    sum_column = img.sum(axis=0).cumsum()
    x_center = -1
    for i in range(sum_column.size):
        if sum_column[i] > sum_point / 2:
            x_center = i
            break
    if x_center < 0:
        message = 'seq:%s, frame:%s, no center.' % (
            '-'.join(seq_info), frame_name)
        warn(message)
        log_print(pid, WARNING, message)
        return None
    h_T_W = int(T_W / 2)
    left = x_center - h_T_W
    right = x_center + h_T_W
    if left <= 0 or right >= img.shape[1]:
        left += h_T_W
        right += h_T_W
        _ = np.zeros((img.shape[0], h_T_W))
        img = np.concatenate([_, img, _], axis=1)
    img = img[:, left:right]
    return img.astype('uint8')


def cut_pickle(seq_info, pid):
    seq_name = '-'.join(seq_info)
    log_print(pid, START, seq_name)
    seq_path = os.path.join(INPUT_PATH, *seq_info)
    # print(seq_path)
    out_dir = os.path.join(OUTPUT_PATH, *seq_info)
    frame_list = os.listdir(seq_path)
    frame_list.sort()
    count_frame = 0
    for _frame_name in frame_list:
        frame_path = os.path.join(seq_path, _frame_name)
        
        img = cv2.imread(frame_path)[:, :, 0]
        img = cut_img(img, seq_info, _frame_name, pid)
        if img is not None:
            # Save the cut img
            save_path = os.path.join(out_dir, _frame_name)
            scisc.imsave(save_path, img)
            count_frame += 1
    # Warn if the sequence contains less than 5 frames
    if count_frame < 5:
        message = 'seq:%s, less than 5 valid data.' % (
            '-'.join(seq_info))
        warn(message)
        log_print(pid, WARNING, message)

    log_print(pid, FINISH,
              'Contain %d valid frames. Saved to %s.'
              % (count_frame, out_dir))

def cut_pickle_ouisir(seq_info, pid,seq_path):
    seq_name = '-'.join(seq_info)
    log_print(pid, START, seq_name)
    # seq_path = os.path.join(INPUT_PATH, *seq_info)
    # print(seq_path)
    out_dir = os.path.join(OUTPUT_PATH, *seq_info)
    frame_list = os.listdir(seq_path)
    frame_list.sort()
    count_frame = 0
    for _frame_name in frame_list:
        frame_path = os.path.join(seq_path, _frame_name)
        # print(frame_path)
        img = cv2.imread(frame_path)[:, :, 0]
        img = cut_img(img, seq_info, _frame_name, pid)
        if img is not None:
            # Save the cut img
            save_path = os.path.join(out_dir, _frame_name)
            scisc.imsave(save_path, img)
            count_frame += 1
    # Warn if the sequence contains less than 5 frames
    if count_frame < 5:
        message = 'seq:%s, less than 5 valid data.' % (
            '-'.join(seq_info))
        warn(message)
        log_print(pid, WARNING, message)

    log_print(pid, FINISH,
              'Contain %d valid frames. Saved to %s.'
              % (count_frame, out_dir))

pool = Pool(WORKERS)
results = list()
# pid = 0





print('OUMVLP')

dall = [
'Silhouette_000-00', 'Silhouette_000-01', 'Silhouette_015-00', 'Silhouette_015-01', 
'Silhouette_030-00', 'Silhouette_030-01', 'Silhouette_045-00', 'Silhouette_045-01',
'Silhouette_060-00', 'Silhouette_060-01', 'Silhouette_075-00', 'Silhouette_075-01', 
'Silhouette_090-00', 'Silhouette_090-01', 'Silhouette_180-00', 'Silhouette_180-01', 
'Silhouette_195-00', 'Silhouette_195-01', 'Silhouette_210-00', 'Silhouette_210-01', 
 'Silhouette_225-00', 'Silhouette_225-01', 'Silhouette_240-00', 'Silhouette_240-01', 
 'Silhouette_255-00', 'Silhouette_255-01', 'Silhouette_270-00', 'Silhouette_270-01'
 ]

#-------------------------------------------------------------------------
for dir1 in dall:
    dsplit = dir1.split('_')[1]
    _view =  dsplit.split('-')[0]
    _seq_type =  dsplit.split('-')[1]
    id_path = INPUT_PATH + dir1+'/'
    id_list = os.listdir(id_path)
    id_list.sort()
    for _id in id_list:
        # print(dsplit,_view,_seq_type,_id)
        pid = int(_id)
        seq_info = [_id, _seq_type, _view]
        out_dir = os.path.join(OUTPUT_PATH, *seq_info)
        # print(out_dir,os.path.isdir(out_dir))
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        seq_path = id_path + _id +'/'
        results.append(
            pool.apply_async(
                cut_pickle_ouisir,
                args=(seq_info, pid,seq_path)))
        sleep(0.02)
#---------------------------------------------------------------------------
#--------------------test-------------------------------------------
# for dir1 in dall:
#     print(dir1)
#     dsplit = dir1.split('_')[1]
#     _view =  dsplit.split('-')[0]
#     _seq_type =  dsplit.split('-')[1]
#     id_path = INPUT_PATH + dir1+'/'
#     id_list = os.listdir(id_path)
#     id_list.sort()
#     count1 = 0
#     count2 = 0
#     for _id in id_list:
#         # print(dsplit,_view,_seq_type,_id)
#         pid = int(_id)
#         seq_info = [_id, _seq_type, _view]
#         out_dir = os.path.join(OUTPUT_PATH, *seq_info) 
#         out_dir1 = INPUT_PATH+'/'+ dir1 +'/' + _id
#         len_OUTPUT_PATH = len(os.listdir(out_dir))
#         len_INPUT_PATH = len(os.listdir(out_dir1))
#         # print(len_OUTPUT_PATH,len_INPUT_PATH) 
#         count1 +=1
#         if len_OUTPUT_PATH == len_INPUT_PATH:
#             count2 +=1
#         else:
#             print(dsplit,_view,_seq_type,_id)
#     print(count1 , count2,count1==count2)


pool.close()
unfinish = 1
while unfinish > 0:
    unfinish = 0
    for i, res in enumerate(results):
        try:
            res.get(timeout=0.1)
        except Exception as e:
            if type(e) == MP_TimeoutError:
                unfinish += 1
                continue
            else:
                print('\n\n\nERROR OCCUR: PID ##%d##, ERRORTYPE: %s\n\n\n',
                      i, type(e))
                raise e
pool.join()
