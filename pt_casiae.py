import os
from scipy import misc as scisc
import cv2
import numpy as np
from warnings import warn
from time import sleep
import argparse
import os.path as osp
from multiprocessing import Pool
from multiprocessing import TimeoutError as MP_TimeoutError
import csv
import xarray as xr
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata
import sys
sys.path.append('/home/linbb/CASIA-E/model/network')
import vgg_c3d
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch.nn.functional as F

IF_LOG = True
LOG_PATH = './pretreatment.log'
WORKERS = 0

START = "START"
FINISH = "FINISH"
WARNING = "WARNING"
FAIL = "FAIL"

T_H = 64
T_W = 64

# T_H = 128
# T_W = 128
def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'

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

def cut_pickle(seq_info, pid,INPUT_PATH_ID_LM,OUTPUT_PATH):
    seq_name = '-'.join(seq_info)
    # seq_path = os.path.join(INPUT_PATH, *seq_info)
    out_dir = os.path.join(OUTPUT_PATH, *seq_info)
    # frame_list = os.listdir(seq_path)
    frame_list = os.listdir(INPUT_PATH_ID_LM)
    frame_list.sort()
    count_frame = 0
    for _frame_name in frame_list:
        # frame_path = os.path.join(seq_path, _frame_name)
        frame_path = os.path.join(INPUT_PATH_ID_LM, _frame_name)
        # print(INPUT_PATH_ID_LM)
        # print(frame_path)
        img = cv2.imread(frame_path)
        # print(len(img),len(img[0]))
        img = cv2.imread(frame_path)[:, :, 0]
        img = cut_img(img, seq_info, _frame_name, pid)
        # img = img[:,10:-10]
        # print(img.shape)
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


def verify_test_gallery():
    path = '/home/linbb/CASIA-E/test_gallery/'
    dir1 = sorted(os.listdir(path))

    print('total=',len(dir1))
    count = 0
    for fn1 in dir1:
        pathabb = path + fn1 +'/'
        dir2 = sorted(os.listdir(pathabb))
        if len(dir2) == 1:
            count = count + 1
        # print(fn1,'-len-',len(dir2))
        # for fn2 in dir2:
        #     pathabb_i = pathabb + fn2
    print('subtotal=',count)

def generate_test_gallery():
    INPUT_PATH = '/home/linbb/CASIA-E/test_gallery/'
    OUTPUT_PATH = '/home/linbb/CASIA-E/64test_gallery/'
    # OUTPUT_PATH = '/home/linbb/CASIA-E/128test_gallery/'

    path = INPUT_PATH
    dir1 = sorted(os.listdir(path))

    print('total=',len(dir1))
    pid = 0
    for fn1 in dir1:
        pid = pid + 1
        INPUT_PATH_ID = path + fn1 +'/'
        dir2 = sorted(os.listdir(INPUT_PATH_ID))
        print('id,luanma=',fn1,dir2)
        seq_info = [fn1, 'NM01', '000']
        for fn2 in dir2:
            INPUT_PATH_ID_LM = INPUT_PATH_ID + fn2+'/'
        out_dir = os.path.join(OUTPUT_PATH, *seq_info)
        os.makedirs(out_dir)
        cut_pickle(seq_info, pid, INPUT_PATH_ID_LM, OUTPUT_PATH)
        # if pid ==1 :
        #     break
    print('subtotal=',pid)
def count_train_gallery():
    INPUT_PATH = '/home/linbb/CASIA-E/64train/'
    dir1 = sorted(os.listdir(INPUT_PATH))
    print('conut id = ', len(dir1))
    # for fn1 in dir1:

def generate_train_gallery():
    INPUT_PATH = '/home/linbb/CASIA-E/train/'
    OUTPUT_PATH = '/home/linbb/CASIA-E/64train/'
    # OUTPUT_PATH = '/home/linbb/CASIA-E/128train/'

    path = INPUT_PATH
    dir1 = sorted(os.listdir(path))

    print('total=',len(dir1))
    pid = 0
    for fn1 in dir1:
        pid = pid + 1
        INPUT_PATH_ID = path + fn1 +'/'
        dir2 = sorted(os.listdir(INPUT_PATH_ID))
        print('id,len(luanma)=',fn1,len(dir2))
        nmcount = 1
        for fn2 in dir2:
            seq_info = [fn1, '', '000']
            strnm = "%02d" % nmcount
            nmcount = nmcount + 1
            seq_info[1] = 'NM'+ strnm
            # print(seq_info)


            INPUT_PATH_ID_LM = INPUT_PATH_ID + fn2+'/'
            out_dir = os.path.join(OUTPUT_PATH, *seq_info)
            os.makedirs(out_dir)
            cut_pickle(seq_info, pid, INPUT_PATH_ID_LM, OUTPUT_PATH)
        # if pid ==1 :
        #     break
    print('subtotal=',pid)

def generate_test_probe():
    INPUT_PATH = '/home/linbb/CASIA-E/test_probe/'
    OUTPUT_PATH = '/home/linbb/CASIA-E/64test_probe/'

    path = INPUT_PATH
    dir1 = sorted(os.listdir(path))

    print('total=',len(dir1))
    pid = 0
    for fn1 in dir1:
        pid = pid + 1
        INPUT_PATH_ID = path + fn1 +'/'
    #     dir2 = sorted(os.listdir(INPUT_PATH_ID))
    #     print('id,len(luanma)=',fn1,len(dir2))
    #     nmcount = 1
    #     for fn2 in dir2:
        seq_info = [fn1]
    #         strnm = "%02d" % nmcount
    #         nmcount = nmcount + 1
    #         seq_info[1] = 'NM'+ strnm
    #         # print(seq_info)


    #         INPUT_PATH_ID_LM = INPUT_PATH_ID + fn2+'/'
        out_dir = os.path.join(OUTPUT_PATH, *seq_info)
        os.makedirs(out_dir)
        cut_pickle(seq_info, pid, INPUT_PATH_ID, OUTPUT_PATH)
        # if pid ==1 :
        #     break
    print('subtotal=',pid)

def img2xarray(flie_path):
    imgs = sorted(list(os.listdir(flie_path)))
    frame_list = [np.reshape(
        cv2.imread(osp.join(flie_path, _img_path)),
        [64, 64, -1])[:, :, 0]
                    for _img_path in imgs
                    if osp.isfile(osp.join(flie_path, _img_path))]
    num_list = list(range(len(frame_list)))
    data_dict = xr.DataArray(
        frame_list,
        coords={'frame': num_list},
        dims=['frame', 'img_y', 'img_x'],
    )
    return data_dict

def extract_test_gallery(modelpath):
    
    print('----extract_test_gallery-----')
    INPUT_PATH = '/home/linbb/CASIA-E/64test_gallery/'
    net = vgg_c3d.c3d_vgg_Fusion(num_classes=500)
    net = nn.DataParallel(net, device_ids=[0])
    net.load_state_dict(torch.load(modelpath))

    label = []
    feature = []
    with torch.no_grad():
        dir1 = sorted(os.listdir(INPUT_PATH))
        pid = 0
        for fn1 in dir1:
            pid = pid + 1
            INPUT_PATH_ID = INPUT_PATH + fn1 +'/'
            # print(INPUT_PATH_ID)
            INPUT_PATH_ID_ALL = INPUT_PATH_ID+'NM01/000/'
            # dir2 = sorted(os.listdir(INPUT_PATH_ID))
            # print(INPUT_PATH_ID_ALL)
            templen = len(os.listdir(INPUT_PATH_ID_ALL))
            if templen==0:
                print(fn1+'=0 len')
                cidxarry = np.zeros(10*64*44).reshape(10,64,44)
            else:
                cidxarry = img2xarray(INPUT_PATH_ID_ALL)[:, :, 10:-10].astype('float32') / 255.0
            # print(cidxarry.shape)
            seq = np.array(cidxarry)
            seq = np.float32(seq)
            seq = torch.from_numpy(seq)
            seq = seq.unsqueeze(0)
            seq = seq.unsqueeze(0)

            # seq = torch.cat((seq,seq,seq),dim=1)

            seq = Variable(seq.cuda())
            #----------offical---------------------
            outputs,_ = net(seq)
            n,_,_ = outputs.size()
            outputs = outputs.view(n,-1)
            # print('ex-',outputs.shape)
            label.append(fn1)
            feature.append(outputs)
            # #----------local---------------------
            # _,_,tall,_,_ = seq.size()
            # # print(seq.shape,tall)
            # ft = 0
            # while (ft+1)*30<=tall:
            #     # print('---',ft+1)
            #     tempseq = seq[:,:,ft*30:(ft+1)*30,:,:]
            #     outputs,_ = net(tempseq)
            #     n,_,_ = outputs.size()
            #     outputs = outputs.view(n,-1)
            #     ft = ft + 1
            #     label.append(fn1)
            #     feature.append(outputs)

            # if pid == 5:
            #     break
            # print(label)
            # print(feature[0].shape,feature[1].shape)
    feature = torch.cat(feature,0)
    # print(feature.shape)
    return label, feature



def cuda_dist(x, y):
    # x = torch.from_numpy(x).cuda()
    # y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist

def testout():
    
    path = '/home/linbb/CASIA-E/64test_probe/'
    modelpath = '/home/linbb/CASIA-E/GL_500_ptoumv_12_8_64f_CASIA-E_500_False_256_0.2_96_full_64-15000-encoder.ptm'
    print(modelpath)
    label, feature_gallery = extract_test_gallery(modelpath)
    print('feature_gallery shape = ',feature_gallery.shape)

    net = vgg_c3d.c3d_vgg_Fusion(num_classes=500)
    net = nn.DataParallel(net, device_ids=[0])
    net.load_state_dict(torch.load(modelpath))
    with torch.no_grad():
        with open('/home/linbb/CASIA-E/demo.csv', 'r') as f:
            reader = csv.reader(f)
            listcsv = []
            for row in reader:
                listcsv.append(row)
                # print(row)
            # print(len(listcsv))
            # print(listcsv[0])
            for i in range(1,len(listcsv)):
                # print(listcsv[i])
                cid = listcsv[i][0]
                pathcid = path + cid+'/'
                templen = len(os.listdir(pathcid))
                if templen==0:
                    print(cid+'=0 len')
                    cidxarry = np.zeros(10*64*44).reshape(10,64,44)
                else:
                    cidxarry = img2xarray(pathcid)[:, :, 10:-10].astype('float32') / 255.0

                # print(cidxarry.shape)
                seq = np.array(cidxarry)
                seq = np.float32(seq)
                seq = torch.from_numpy(seq)
                seq = seq.unsqueeze(0)
                seq = seq.unsqueeze(0)

                # seq = torch.cat((seq,seq,seq),dim=1)



                seq = Variable(seq.cuda())
                # print(seq.shape)
                outputs,_ = net(seq)
                n,_,_ = outputs.size()
                outputs = outputs.view(n,-1)
                # print(outputs.shape,feature_gallery.shape)

                dist = cuda_dist(outputs, feature_gallery)
                # print(dist)
                idx = dist.sort(1)[1].cpu().numpy()[0]
                # print(idx)
                # print(idx[0],label[idx[0]])
                # revise label
                listcsv[i][1] = int(label[idx[0]])
                print('id=',i,'  ',listcsv[i][0],'  ',listcsv[i][1])
                # if i == 5:
                    # break
                


        with open('/home/linbb/CASIA-E/submission.csv', 'w', newline='') as csvfile:
            writer  = csv.writer(csvfile)
            for row in listcsv:
                writer.writerow(row)

if __name__ == "__main__":
    # generate_test_gallery()
    # generate_train_gallery()
    # generate_test_probe()
    # veritycsv()
    # count_train_gallery()
    testout()