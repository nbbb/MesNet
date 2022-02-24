#!/usr/bin python3
# -*- encoding: utf-8 -*-
# @Author : 犇犇
# @File : com_h5.py
# @Time : 2022/2/21 9:03
import h5py
import os
train_data=h5py.File("/data2/ntz/ntz_code2/Data/MSRVTT/phase_feats/MSR-VTT_InceptionV4_s3_train.hdf5",'r')
test_data=h5py.File("/data2/ntz/ntz_code2/Data/MSRVTT/phase_feats/MSR-VTT_InceptionV4_s3_test.hdf5",'r')
val_data=h5py.File("/data2/ntz/ntz_code2/Data/MSRVTT/phase_feats/MSR-VTT_InceptionV4_s3_val.hdf5",'r')
out_file="/data2/ntz/ntz_code2/Data/MSRVTT/Feats/MSR-VTT_InceptionV4_s3.hdf5"
h5 = h5py.File(out_file, 'w') if not os.path.exists(out_file) else h5py.File(out_file, 'r+')
for vid in train_data.keys():
    if vid in h5.keys():
        print(vid)
        continue
    h5[vid] = train_data[vid][()]
for vid in test_data.keys():
    if vid in h5.keys():
        print(vid)
        continue
    h5[vid] = test_data[vid][()]
for vid in val_data.keys():
    if vid in h5.keys():
        print(vid)
        continue
    h5[vid] = val_data[vid][()]
h5.close()
test_data.close()
train_data.close()
val_data.close()
