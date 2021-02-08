import cv2
import numpy as np
from os.path import join
from itertools import count
import os


def get_FPS(video):
    """
    Get video FPS
    """
    vidcap = cv2.VideoCapture(video)
    FPS = vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.release()
    print(f'This video has {FPS:.2f} fps.')
    return FPS


def read_images(video):
    """
    Returns:
        dir: image_name: cv2.imread(image)
    """
    vidcap = cv2.VideoCapture(video)
    success, img = vidcap.read()
    frame_id = count()
    images = {}
    while success:
        the_id = str(next(frame_id)).zfill(7)
        img = img[:, -565:, :]
        images[f'frame_{the_id}.png'] = img
        success, img = vidcap.read()
    vidcap.release()
    return images


def draw_box(img, infos, THRESHOLD, basic_box=False, det=True):
    """
    Args:
        infos (list[list]): [[x1,y1,x2,y2,catId,score,track_id]]
    """
    template = "{}: {:.2f}"
    cla_id = {
            1: 'primary',
            2: 'elevated',
            3: 'ulcer',
            4: 'polyp',
            5: 'tumor',
            6: 'erosion'
        }
        
    Color = {
        '0': (255, 255, 255),
        '1': (255, 255, 0),
        '2': (255, 0, 255)
    }
    flag = False
    h, w, _ = img.shape
    for info in infos:
        if info[:4] == [0]*4:
            continue
        x1, y1, x2, y2, catId, score, track_id = info
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        width, height = x2-x1, y2-y1
        if width <= 0 or height <= 0:
            continue
        if basic_box and os.path.exists(basic_box):
            basic_b = cv2.imread(basic_box)
            basic_b = cv2.resize(basic_b, (width, height))


        if float(score) >= THRESHOLD:
            if basic_box:
                assert (img[y1:y2,x1:x2,].shape == basic_b.shape)
                merge = cv2.addWeighted(img[y1:y2,x1:x2,], 0.9, basic_b, 1.0, 0)
                img[y1:y2,x1:x2,] = merge
            else:
                color = Color[str(detected)[0]]
                cv2.rectangle(img, (int(float(x1)), int(float(y1))), (int(float(x2)), int(float(y2))), color, 2)
            cla = f'{str(track_id).zfill(3)} {cla_id[int(catId)]}: {float(score):.2}'
            cv2.putText(img, cla, (int(float(x1)), int(float(y1))), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            flag = True
    return img, flag