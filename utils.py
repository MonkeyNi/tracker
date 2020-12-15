import cv2
import os
import collections
import numpy as np
import time
from visualization import draw_box


def createTrackerByName(trackerType):
    trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]: 
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are: {}'.format(trackerTypes))
    return tracker


def cor_change(box):
    """
    From x,y,w, h to x,y,x,y
    """
    box = [int(x) for x in list(box)]
    x1, y1, w, h = box
    return [x1, y1, w, h]


def get_infos(images, infer_txt):
    """
    Initialize detection results from detection txt file
    For each line, it should have format:
        "x1, y1, x2, y2, cla_id, scores\n"

    Args:
        images: list of image names
        infer_txt: path to detection result txt files

    Returns:
        dir: {image_name: [[x1,y1,x2,y2,cla_id,score,detected]]} 
            detected:
                0: detected
                1: detected and tracked
                2: not detected but tracked
    """
    res = collections.defaultdict(list)
    
    with open(infer_txt) as f:
        lines = f.readlines()
        for line in lines:
            if not line: continue
            infos = line.split(',')
            f_name = infos[0] if infos[0].endswith('png') else f'{infos[0]}.png'
            info = infos[1:]
            score = float(info[5])
            info = [int(float(x)) for x in info]
            info[5] = score
            if len(info) != 7:
                info = info + [0]  # detect status
            res[f_name].append(info)
        # add FN
        base = [0]*4 + [1, 0.1, 0]
        for im in images:
            if len(res[im]) == 0:
                res[im] = [base]
    return {im: res[im] for im in sorted(res.keys())}


def get_track_bboxes(frame, tracker_pool):
    """
    Get track prediction for a frame

    Args:
        frame (np.array): the frame
        tracker_pool: pool of Trackers

    Returns:
        tracked_boxes: [[tracked_box]]
        tracked_time: only useful during testing
    """
    tracked_boxes, tracked_time = [], []
    for tracker in tracker_pool.trackers:
        track = tracker.tracker
        # import pdb; pdb.set_trace()
        start = time.time()
        success, track_boxes = track.update(frame)
        end = time.time()
        tracked_time.append((end-start))
        if (np.array(track_boxes) >= 0).min():
            tracked_boxes.extend(track_boxes)
    tracked_boxes = [cor_change(list(box)) for box in tracked_boxes]
    return tracked_boxes, tracked_time


def linear_assignment(cost_matrix):
    """
    Data association
    TODO: Hungarian algorithm is used now. It does not have weight. 
          KM or feature related algorithm maybe used in the future
    """
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(det_bboxes, track_bboxes):
    """
    Compute overlap between detection results and tracked results

    Args:
        det_bboxes ([[]])
        track_bboxes ([[]]])
    """
    det_bboxes = np.expand_dims(det_bboxes, axis=1)
    track_bboxes = np.expand_dims(track_bboxes, axis=0)

    x1 = np.maximum(det_bboxes[..., 0], track_bboxes[..., 0])
    y1 = np.maximum(det_bboxes[..., 1], track_bboxes[..., 1])
    x2 = np.minimum(det_bboxes[..., 2], track_bboxes[..., 2])
    y2 = np.minimum(det_bboxes[..., 3], track_bboxes[..., 3])
    w = np.maximum(0., x2 - x1)
    h = np.maximum(0., y2 - y1)
    wh = w * h
    overlap = wh / ((det_bboxes[..., 2] - det_bboxes[..., 0]) * (det_bboxes[..., 3] - det_bboxes[..., 1])
        + (track_bboxes[..., 2] - track_bboxes[..., 0]) * (track_bboxes[..., 3] - track_bboxes[..., 1]) - wh)
    return overlap


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if (len(trackers) == 0):
        return np.empty((0,2), dtype=int), np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    
    iou_matrix = iou_batch(detections, trackers)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        # one for one
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty((0, 2))
    
    unmatched_detections = [i for i, _ in enumerate(detections) if not i in matched_indices[:, 0]]
    unmatched_trackers = [i for i, _ in enumerate(trackers) if not i in matched_indices[:, 1]]

    # match_2 used to update lesion tracker with the lasted detection result
    matches, matches_2 = [],[]
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
            if iou_matrix[m[0], m[1]] >= min(iou_threshold*2, 0.5):
                matches_2.append(m.reshape(1, 2))
    matches = np.empty((0,2), dtype=int) if len(matches) == 0 else np.concatenate(matches, axis=0)
    matches_2 = np.empty((0,2), dtype=int) if len(matches_2) == 0 else np.concatenate(matches_2, axis=0)
    return matches, matches_2, np.array(unmatched_detections), np.array(unmatched_trackers)


def generate_tracked_info(t_bboxes, tracker_pool, GT_THRESHOLD):
    """
    Return tracked info which will be saved in output txt file
    """
    tracked_info = []
    for tb, t in zip(t_bboxes, tracker_pool.trackers):
        info = tb + [1, t.score, 2]
        tracked_info.append(info)
    return tracked_info


def nms(dets, thresh=0.4):
    dets = np.array(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def compute_intersect(rec1, rec2):
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    sum_area = S_rec1 + S_rec2

    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect


def nms_2(infos, thresh=0.9):
    """
    This function is used to deal with small/big overlap problem
    """
    if len(infos) == 1:
        return []
    infos = np.array(infos)
    dets, clas, scores = infos[:,:4], infos[:,4], infos[:,5]
    areas = np.array([(x2-x1)*(y2-y1) for x1,y1,x2,y2 in dets])
    inds = areas.argsort()
    dets = dets[inds]
    delete = []
    for i, box in enumerate(dets):
        flag = False
        check = dets[i+1:]
        for j, che in enumerate(check, i+1):
            intersect = compute_intersect(box, che)
            if intersect == 0:
                continue
            if float(intersect)/areas[inds[i]] >= thresh and clas[inds[i]] == clas[inds[j]]:
                flag = True
                break
        if not flag:
            delete.append(inds[i])
    return delete


def save_key_frames(key_frames, out_folder, basic_box=False):
    """
    Save key frames

    Args:
        key_frames ([list]): [[name, frame], [[box], score, cat_id]]
    """
    if len(key_frames) == 0:
        return
    for key_f in key_frames:
        img_name, img = key_f[0]
        infos = key_f[1][0] + [key_f[1][2]] + [key_f[1][1]] + [1]
        img, _ = draw_box(img, [infos], 0, basic_box=basic_box)
        img_name = str(key_f[1][2]) + '_' + img_name
        cv2.imwrite(os.path.join(out_folder, img_name), img)
