
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

def video_to_images():
    import subprocess

    base_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/video_normal'
    dst_path = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/video_normal/image_normal'
    for file_name in os.listdir(base_folder):
        if not ('.avi' in file_name or '.mp4' in file_name):
            continue

        name, ext = os.path.splitext(file_name)

        dst_directory_path = os.path.join(dst_path, name)
        print(dst_directory_path)
        if not os.path.exists(dst_directory_path):
            os.mkdir(dst_directory_path)

        video_file_path = os.path.join(base_folder, file_name)
        cmd = 'ffmpeg -i \"{}\" \"{}/image_%05d.jpg\"'.format(video_file_path, dst_directory_path)
        print(cmd)
        subprocess.call(cmd, shell=True)


def create_TrainTestlist(clip_folder, TrainTest_folder):

    label_map = {'others':'1', 'pick':'2', 'scratch':'3'}

    TrainTestlist = {'others':[], 'pick':[], 'scratch':[]}
    for sub in ['others', 'pick', 'scratch']:
        sub_folder = clip_folder + sub
        count = 0
        for i, subsub in enumerate(sorted(os.listdir(sub_folder))):
            contents = sub + '/' + subsub + ' ' + label_map[sub] + '\n'
            TrainTestlist[sub].append(contents)
            count += 1

        random.shuffle(TrainTestlist[sub])
        print(sub, count)

    num_test_for_each_cls = 40
    num_list_for_train = 6
    num_exam = 1000

    ## get TEST list
    Testlist = []
    for sub in ['others', 'pick', 'scratch']:
        Testlist += TrainTestlist[sub][:num_test_for_each_cls]
    random.shuffle(Testlist)
    print('testlist01.txt', len(Testlist))

    Testlist_name = TrainTest_folder + 'testlist01.txt'
    Testlist_file = open(Testlist_name, 'w')
    for contents in Testlist:
        Testlist_file.write(contents)
    Testlist_file.close()


    ## get TRAIN list
    for i in range(num_list_for_train):

        Trainlist = TrainTestlist['others'][num_test_for_each_cls+i*num_exam:num_test_for_each_cls+(i+1)*num_exam]
        for sub in ['pick', 'scratch']:
            Trainlist += TrainTestlist[sub][num_test_for_each_cls:]
        random.shuffle(Trainlist)
        print('trainlist0{}.txt'.format(i), len(Trainlist))

        Trainlist_name = TrainTest_folder + 'trainlist0{}.txt'.format(i)
        Trainlist_file = open(Trainlist_name, 'w')
        for contents in Trainlist:
            Trainlist_file.write(contents)
        Trainlist_file.close()



def create_Testlist(clip_folder, TrainTest_folder):

    Testlist_name = TrainTest_folder + '/testlist01.txt'

    Testlist = open(Testlist_name, 'w')
    print(clip_folder)
    for sub in sorted(os.listdir(clip_folder)):
        sub_folder = clip_folder + sub
        for subsub in sorted(os.listdir(sub_folder)):
            contents = sub + '/' + subsub + ' 0' + '\n'
            Testlist.write(contents)

    Testlist.close()


def remove_bad_others():
    ske_folder = 'hand_static_BG'

    base_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/'
    __action__ = ['others', 'pick', 'scratch']
    bad_others_list = ['_'.join(line.split('_')[1:5]) for line in open(os.path.join(base_folder, 
        'bad_others', 'bad_others.txt'), 'r')]
    # print(bad_others_list)

    for act in __action__:

        if act != 'others':
            continue

        base_in_clip_folder = base_folder + act + '/clips/'

        for sub_id, sub in enumerate(os.listdir(base_in_clip_folder)):

            # print(sub)
            if sub in bad_others_list:
                print('bad_others_list', sub)
                # print(os.path.join(base_in_clip_folder, sub), os.path.join(base_folder, 'bad_others', sub))
                # os.rename(os.path.join(base_in_clip_folder, sub), os.path.join(base_folder, 'bad_others', sub))
                continue


def vis_clip():
    base_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/hand_static_BG/'
    # base_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/video_normal'
    # __action__ = ['others', 'pick', 'scratch']

    act = 'scratch' #'others' #'clip_scratch' 

    dst_folder = os.path.join(base_folder, act)
    for sub in os.listdir(dst_folder):
        if 'others' in sub:
            continue

        sub_folder = os.path.join(dst_folder, sub)
        heatmap_name_list = [img_name for img_name in os.listdir(sub_folder) if img_name.endswith('.jpg') and 'heatmap' in img_name]
        image_name_list = [img_name for img_name in os.listdir(sub_folder) if img_name.endswith('.jpg') and not 'heatmap' in img_name]

        N1 = len(heatmap_name_list)
        N2 = len(image_name_list)
        assert(N1 == N2)

        print(sub_folder)
        for i in range(N1 * 100):
            #mage_path = os.path.join(sub_folder, 'image_{:05d}.jpg'.format(i%N1+1))
            image_path = os.path.join(sub_folder, 'image_{:05d}_heatmap.jpg'.format(i%N1+1))
            img = cv2.imread(image_path)
            cv2.imshow('img', img)
            if cv2.waitKey(15) == ord('a'):
                break

if __name__ == "__main__":

    vis_clip()

    # video_to_images()

    # clip_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/hand_static_BG/'
    # TrainTest_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/TrainTestlist/hand_static_BG/'
    # create_TrainTestlist(clip_folder, TrainTest_folder)
   