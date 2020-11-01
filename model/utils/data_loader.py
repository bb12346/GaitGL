import os
import os.path as osp

import numpy as np

from .data_set import DataSet


def load_data(dataset_path, resolution, dataset, pid_num, pid_shuffle, cache=True):
    seq_dir = list()
    view = list()
    seq_type = list()
    label = list()

    for _label in sorted(list(os.listdir(dataset_path))):

        label_path = osp.join(dataset_path, _label)
        for _seq_type in sorted(list(os.listdir(label_path))):
            seq_type_path = osp.join(label_path, _seq_type)
            for _view in sorted(list(os.listdir(seq_type_path))):
                _seq_dir = osp.join(seq_type_path, _view)
                seqs = os.listdir(_seq_dir)
                if len(seqs) > 0:
                    seq_dir.append([_seq_dir])
                    label.append(_label)
                    seq_type.append(_seq_type)
                    view.append(_view)

    if dataset == 'CASIA-B' and dataset == 'CASIA-E':
        pid_fname = osp.join('partition', '{}_{}_{}.npy'.format(
            dataset, pid_num, pid_shuffle))
        # print(pid_fname)
        if not osp.exists(pid_fname):
            pid_list = sorted(list(set(label)))
            if pid_shuffle:
                np.random.shuffle(pid_list)
            pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
            os.makedirs('partition', exist_ok=True)
            np.save(pid_fname, pid_list)
        # pid_list = np.load(pid_fname)
        pid_list = np.load(pid_fname, allow_pickle=True)
        train_list = pid_list[0]
        test_list = pid_list[1]
    else:
        pid_fname = osp.join('partition', '{}_{}_{}.npy'.format(
            dataset, pid_num, pid_shuffle))
        # print(pid_fname)
        pid_list = list()
        if not osp.exists(pid_fname):
            for i in range(1,10307):
                if i % 2!=0:
                    # counteven += 1 
                    tem = '%05d'%i
                    pid_list.append(tem)  
            for i in range(1,10307):
                if i % 2==0:
                    # countodd += 1 
                    tem = '%05d'%i
                    pid_list.append(tem)
            pid_list.append('10307')
            # pid_list = sorted(list(set(label)))
            if pid_shuffle:
                np.random.shuffle(pid_list)
            pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
            os.makedirs('partition', exist_ok=True)
            np.save(pid_fname, pid_list)
        # pid_list = np.load(pid_fname)
        pid_list = np.load(pid_fname, allow_pickle=True)
        train_list = pid_list[0]
        test_list = pid_list[1]
        # print(len(train_list))
    print('lentestdata--',len(train_list),len(test_list))
    train_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in train_list],
        [label[i] for i, l in enumerate(label) if l in train_list],
        [seq_type[i] for i, l in enumerate(label) if l in train_list],
        [view[i] for i, l in enumerate(label)
         if l in train_list],
        cache, resolution, cut=True)
    test_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in test_list],
        [label[i] for i, l in enumerate(label) if l in test_list],
        [seq_type[i] for i, l in enumerate(label) if l in test_list],
        [view[i] for i, l in enumerate(label)
         if l in test_list],
        cache, resolution, cut=True)
    print('len train,test--',len(train_source),len(test_source))
    # print(train_source[0])
    # print(test_source[0])
    return train_source, test_source
