# coding=utf-8
import csv
import os


src_dir = 'MPV/all'
dst_dir = 'MPV/data'
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir.decode('utf-8'))
with open('MPV/all_poseA_poseB_clothes_0607.csv') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        person1_dir = row[0]
        person2_dir = row[1]
        cloth_dir = row[2]
        is_train = row[3] == 'train'

