
from detection import detection

import os
import sys
import cv2


if __name__ == "__main__":

    T = 45
    reg_model_file = 'results-scratch-18-static_BG/save_200.pth'
    # reg_model_file = 'results-scratch-18/save_200.pth'

    skeleton_opt = 'Openpose' #'Alphapose' #'Openpose'
    is_static_BG = True
    video_name = '1_7'
    detection = detection(reg_model_file, skeleton_opt=skeleton_opt, is_vis=False, is_static_BG=is_static_BG, thres=0.0)

    video_path = '/media/qcxu/qcxuDisk/windows_datasets_all/videos_test/small_videos/'
    input_video_name = video_path+video_name+'.avi'
    print(input_video_name)
    stream = cv2.VideoCapture(input_video_name)
    width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path+video_name+str(is_static_BG)+'.avi', fourcc, 25.0, (width, height))

    ret = True
    frame_id = 0
    img_out_all = None
    imglist = []
    while ret:
        (ret, frame) = stream.read()
        frame_id += 1

        if frame_id <= T:
            imglist.append(frame)
            if frame_id == T:
                frame_id = 0
                img_out_all = detection.run(imglist)
                imglist = []
            else:
                if img_out_all is None:
                    continue
                else:
                    out.write(img_out_all[frame_id-1])

                    # cv2.imshow('skeletons', img_out_all[frame_id-1])
                    # cv2.waitKey(15)

    out.release()
    cv2.destroyAllWindows()
    detection.print_runtime()
