
from Detection import Detection

import os
import sys
import cv2
import time


def demo_video(is_save_avi=False, is_static_BG=True):

    is_heatmap = True

    T = 30
    if is_heatmap:
        reg_model_file = 'results-scratch-18-static_BG-30-skeleton-concatenate-iter6/save_300.pth'
        model_type = 'resnet_skeleton'
        thres = 0.8
    else:
        reg_model_file = 'results-scratch-18-static_BG-30-iter5/save_200.pth'
        model_type = 'resnet'
        thres = 0.5

    skeleton_opt = 'Alphapose' #'Alphapose' #'Openpose'

    det = Detection(reg_model_file, model_type, skeleton_opt, cuda_id_list=[1,0],
        sample_duration=T, sample_rate=1, is_static_BG=True, is_heatmap=is_heatmap, thres=thres)

    # video_path = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/video_scratch/'
    video_path = '/media/qcxu/qcxuDisk/windows_datasets_all/videos_test'
    # video_name = '192.168.1.102_01_20190117204506847.mp4'
    out_path = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/videos_test/result_all_hand_concatenate'
    for video_name in os.listdir(video_path):
        if not '.mp4' in video_name and not '.webm' in video_name and not '.avi' in video_name:
            continue

        # if video_name != '2019-01-22-192445.webm':
        #     continue
    
        input_video_name = os.path.join(video_path, video_name)
        print('\nreading video from file ' + input_video_name + '\n')
        if not os.path.exists(input_video_name):
            print('Error! No such video exists')
            return
        stream = cv2.VideoCapture(input_video_name)

        width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fixed_width = 480
        height = int(fixed_width/width*height)
        width = fixed_width

        out = None
        if is_save_avi:
            out_video_name = os.path.join(out_path, '.'.join(video_name.split('.')[:-1]) + '_'+ skeleton_opt + '.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(out_video_name, fourcc, 25.0, (width, height))

        time1 = time.time()

        frame_id = 0
        frame_show_id = 0
        img_out_all = None
        imglist = []
        while True:
            (ret, frame) = stream.read()
            if not ret:
                break

            frame = cv2.resize(frame, (width, height))
            cv2.putText(frame, str(frame_show_id), (width-80, 25), 
                cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

            frame_id += 1
            frame_show_id += 1

            if frame_id <= T:
                imglist.append(frame)
                if frame_id == T:
                    frame_id = 0
                    img_out_all = det.run(imglist, out_path+str(frame_show_id))
                    imglist = []
                else:
                    if not is_save_avi or img_out_all is None:
                        continue
                    else:
                        out.write(img_out_all[frame_id-1])

        det.print_runtime()

        if is_save_avi:
            out.release()
        cv2.destroyAllWindows()

        print('total time = ', time.time() - time1)
        print('total frame = ', frame_show_id)
        print('process fps = ', frame_show_id / (time.time() - time1))



if __name__ == "__main__":

    demo_video(is_save_avi=True, is_static_BG=True)

