import cv2
import os
from os.path import join
import argparse
import numpy as np
import time
from utils import get_infos, save_key_frames, get_id
from visualization import read_images, get_FPS, draw_box
from tracker_eco import Tracker, Tracker_pool, Tracking
from itertools import count


def create_video(imgs, video_out='ori_tracked.avi'):
    """
    Create detection video and tracked video for compariation.
    Left is basic detection result and right is tracked result
    """
    print(f'There are {len(imgs)} frames in this video.')
    assert (len(imgs) != 0)
    h, w, c = list(imgs.values())[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = FPS
    # w = 565
    out = cv2.VideoWriter(join(out_folder, video_out), fourcc, fps, (2*w, h))  
    # for testing
    out_images = join(out_folder, video_out)
    out_images = out_images.replace('.avi', '')
    if not os.path.exists(out_images):
        os.makedirs(out_images)
    # detection results
    detect_infos = get_infos(list(imgs.keys()), join(infer_path, infer_txt))
    # tracked results
    track_infos = get_infos(list(imgs.keys()), join(out_folder, track_out))

    p_count = [0, 0]
    for im in imgs:
        tmp = imgs[im].copy()
        ims = [imgs[im], tmp]
        d_info, t_info = detect_infos[im], track_infos[im]
        blank = np.zeros([h, 2*w, c], np.uint8)
        for i, infos in enumerate([d_info, t_info]):
            if set(infos[0][:4]) != {0}:
                img, flag = draw_box(ims[i], infos, GT_THRESHOLD, basic_box=basic_box)
                if flag: p_count[i] += 1
                blank[:, i*w:(i+1)*w, :] = img
            else:
                blank[:, i*w:(i+1)*w, :] = imgs[im]
        out.write(blank)
        # cv2.imwrite(join(out_images, im), blank)
    print(f'After second threshold: {GT_THRESHOLD}, detected lesion increase from {p_count[0]} to {p_count[1]}.')
    out.release()


def Track():
    # save tracked output
    tract_out = open(join(out_folder, track_out), 'w')
    tracker_pool = Tracker_pool()
    track_time, end_to_end_time, trackers = [], [], []

    images = get_infos(list(basic_images.keys()), join(infer_path, infer_txt))
    end_image = get_id(list(basic_images.keys())[-1])
    only_T = 0
    track_id = count(start=1)
    for image in images:
        print(f'\rCurrent processing frame is: {image}', end="")
        frame = basic_images[image]
        infos = images[image]
        trackers.append(len(tracker_pool.trackers))
        start = time.time()
        tracking = Tracking(image,
                            frame,
                            infos,
                            tracker_pool,
                            tracked_threshold=t_threshold,
                            gt_threshold=GT_THRESHOLD,
                            TrackerType=TrackerType,
                            track_id=track_id)
        tracked_infos, tracker_pool, tracked_time, key_frames, only_tracked = tracking.update()
        gap = time.time() - start
        end_to_end_time.append(gap)
        track_time.extend(tracked_time)
        only_T += only_tracked
        # save key frames
        save_key_frames(key_frames, key_frame_out, basic_box=basic_box)
        # record tracked result to txt file
        for info in tracked_infos:
            tract_out.write(image + ',')
            for ele in info[:-1]:
                tract_out.write(str(ele) + ',')
            tract_out.write(str(info[-1]) + '\n')
    tract_out.close()
    # save still alive tracker key frames
    key_frames = [t.key_frame + [t.key_start, t.track_id] for t in tracker_pool.trackers]
    key_frames.append(end_image)
    save_key_frames(key_frames, key_frame_out, basic_box=basic_box)

    if len(track_time) == 0:
        print('\nThere is no detection results.')
    else:
        print('\nAverage tracker speed is {}'.format(sum(track_time)/len(track_time)))
        print('Average end to end time is {}. Max is {}.'.format(sum(end_to_end_time)/len(end_to_end_time), max(end_to_end_time)))
        print('There are {} frames are tracked only.'.format(only_T))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", help='vidoe used to test', type=str, default='')
    parser.add_argument("-d", "--detection_result", help='path to detection result', type=str, default='')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # video is used to get FPS and basic image information
    video = args.video
    basic_images = read_images(video)  # image_name: img

    # detection result and where tracked result will be saved
    infer_path = args.detection_result
    infer_path, infer_txt = os.path.split(infer_path)


    basic_box = join(os.getcwd(), 'bbox.png')
    trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT', 'ECO']
    TrackerType = trackerTypes[-1]
    print(f'Current tracking algorithm is {TrackerType}.')
    EXTENSINO, GT_THRESHOLD, t_threshold, V = 'png', 0.3, 0.1, '0.7'

    out_folder = join(infer_path, f'v{V}_{TrackerType}_{GT_THRESHOLD}_{t_threshold}')
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    key_frame_out = join(out_folder, 'key_frames')
    if not os.path.exists(key_frame_out):
        os.makedirs(key_frame_out)

    track_out = f'track_{infer_txt}'
    Track()

    FPS = get_FPS(video)  # get the FPS

    video_out = (f'{infer_txt}').replace('txt', 'avi')
    create_video(basic_images, video_out=video_out)