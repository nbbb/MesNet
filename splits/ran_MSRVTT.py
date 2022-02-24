#!/usr/bin python3
# -*- encoding: utf-8 -*-
# @Author : 犇犇
# @File : ran_MSRVTT.py
# @Time : 2022/2/21 9:22
import json
import random
def load_splits():
    with open('../data/MSR-VTT/metadata/train.list', 'r') as fin:
        train_vids = json.load(fin)
    with open('../data/MSR-VTT/metadata/valid.list', 'r') as fin:
        val_vids = json.load(fin)
    with open('../data/MSR-VTT/metadata/test.list', 'r') as fin:
        test_vids = json.load(fin)
    all_vids=train_vids+val_vids+test_vids

    return all_vids


def save_list(vids,phase):
    with open('../data/MSR-VTT/metadata/{}_r3.list'.format(phase), 'w') as fin:
        json.dump(vids,fin)
    fin.close()

if __name__ == "__main__":
    all_vids=load_splits()
    random.shuffle(all_vids)
    save_list(all_vids[:6513],"train")
    save_list(all_vids[6513:7010], "valid")
    save_list(all_vids[7010:], "test")

