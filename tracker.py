import cv2
from utils import createTrackerByName, get_track_bboxes, generate_tracked_info
from utils import iou_batch, associate_detections_to_trackers, nms, nms_2, get_id
from utils import get_roi
from matching import cosine_distance
import numpy as np


class Tracker():
    def __init__(self, infos, trackerType, age=2):
        """
        Initialize a tracker with frame and detection information.
        Args:
                infos ([list]): [frame ([image_name, np.array]), [boxes, scores, cats]].
                trackerType: algorithm used to track
                age (int, optional): to record the tracker status. Defaults to 2.
        """
        self.frame_name = infos[0][0]
        self.frame = infos[0][1]
        self.bbox = infos[1][0]
        self.score = infos[1][1]
        self.cat = infos[1][2]
        self.trackerType = trackerType
        self.age = age
        # init single tracker
        self.tracker = createTrackerByName(self.trackerType)
        self.tacker_bool = self.tracker.init(self.frame, tuple(self.bbox))
        self.tracked_pool = [infos[1][:-1]]
        self.tracked_frame = [self.frame_name, self.frame_name]
        # key frame
        start = get_id(self.frame_name)
        self.key_frame = infos
        self.key_f_score = self.key_frame[1][1]
        self.key_start = start
    
    def get_smoothed_cors(self, n, extra_box=False):
        """
        Average previous 5 dets result as showed bounding box
        """
        tracked_pool = self.tracked_pool
        if extra_box:
            tracked_pool.append(extra_box)
        avg_bboxes = np.asarray([b for b, _ in tracked_pool[-n:]])
        return list(np.average(avg_bboxes, axis=0).astype(np.int64))
    
    def update(self, old):
        """
        Used to update lesion tracker with lastest detection result
        Args:
            old (Tracker): old tracker keeps key frame information
        """
        self.tracked_pool = old.tracked_pool
        # record start
        self.key_start = old.key_start
        # update key frame
        if self.key_f_score < old.key_f_score:
            self.key_frame = old.key_frame
            self.key_f_score = old.key_f_score


class Tracker_pool():
    """
    List of trackers. 
    """
    def __init__(self):
        self.trackers = []

    def add(self, tracker):
        self.trackers.append(tracker)

    def delete(self, i):
        self.trackers.pop(i)

    def update(self, i, state=0):
        self.trackers[i].age = min((self.trackers[i].age + state), 2)

    def update_all(self, maximum_trackers=5):
        # get key frames
        key_trackers = [t for t in self.trackers if t.age < 0]
        key_frames = [t.key_frame + [t.key_start] for t in key_trackers]
        
        self.trackers = [t for t in self.trackers if t.age >= 0]
        self.trackers = self.trackers[-maximum_trackers:]
        return key_frames

    def update_miss(self):
        if self.empty():
            return
        for i in range(len(self.trackers)):
            self.trackers[i].age -= 1

    def empty(self):
        return True if len(self.trackers) == 0 else False
    
    def update_match(self, i, infos, max_length=5):
        """
        Args:
            infos ([[]]]): [[box], score, catid]
        """
        frame, info = infos
        info = info[:-1]
        if info not in self.trackers[i].tracked_pool:
            self.trackers[i].tracked_pool.append(info)
            self.trackers[i].tracked_frame[-1] = frame[0]
            self.trackers[i].tracked_pool = self.trackers[i].tracked_pool[-max_length:]
        
    def update_poolTracker(self, m, infos, t_Type):
        """
        Use the lastest detection result init tracker

        Args:
            m ([]): index
            infos ([type]): used to init tracker
            t_Type ([type]): tracker type
        """
        updated_tracker = Tracker(infos, t_Type)
        updated_tracker.update(self.trackers[m[1]])
        self.trackers[m[1]].tracker.clear()
        self.trackers[m[1]] = updated_tracker


class Tracking():
    def __init__(self,
                 image,
                 frame,
                 infos,
                 tracker_pool,
                 tracked_threshold=0.2,
                 gt_threshold=0.3,
                 TrackerType='MEDIANFLOW'):
        """
        Do tracking for single frame

        Args:
            image (str): frame name
            frame (np.array): current frame
            infos ([[]]): [[x1,y1,x2,y2,cla_id,score,detected]]
            tracker_pool: Trackers
            tracked_threshold: threshold determine if tracked

        Returns:
            tracked_info: [[x1,y1,x2,y2,cla_id,score,detected]]
            tracer_pool: updated tracker pool
            tracked_time: []
        """
        self.image = image
        self.frame = frame
        self.infos = infos
        self.tracker_pool = tracker_pool
        self.tracked_threshold = tracked_threshold
        self.tracked_time = []
        self.gt_threshold = gt_threshold
        self.t_Type = TrackerType

    def update(self):
        infos, tracker_pool, frame = self.infos, self.tracker_pool, self.frame
        tracked_info = [[t.bbox, t.age] for t in tracker_pool.trackers]  # for test
        GT_THRESHOLD = self.gt_threshold
        only_tracked = 0  # for test
        if not infos[0][:4] == [0]*4:
            bboxes, cats, scores = [i[:4] for i in infos], [i[4] for i in infos], [i[5] for i in infos]
            ## First: if track is null, just add frame and bbox to tracker
            if tracker_pool.empty():
                for i, box in enumerate(bboxes):
                    score, catid = float(infos[i][-2]), infos[i][-3]
                    if score >= GT_THRESHOLD:
                        tracker_pool.add(Tracker([[self.image, frame], [box, score, catid]], self.t_Type))
            ## Second: track frame, update tracker pool
            else:
                tracked_boxes, self.tracked_time = get_track_bboxes(frame, tracker_pool)
                matched, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(
                    bboxes, tracked_boxes, iou_threshold=self.tracked_threshold
                )
                
                def tmp_infos(ind):
                    return [[self.image, frame], [bboxes[ind], scores[ind], cats[ind]]]

                # update detection score
                for m in matched:
                    infos[m[0]][-2] = min((infos[m[0]][-2] + GT_THRESHOLD + 0.01), 0.99)
                    infos[m[0]][-1] = 1
                    tracker_pool.update(m[1], state=1)
                    # compute smooth cors
                    tracker_pool.update_match(m[1], tmp_infos(m[0]))
                    smoothed_cors = tracker_pool.trackers[m[1]].get_smoothed_cors(5)
                    infos[m[0]][:4] = smoothed_cors
                    # update trackers
                    if scores[m[0]] >= GT_THRESHOLD:
                        tracker_pool.update_poolTracker(m, tmp_infos(m[0]), self.t_Type)
                for m in unmatched_detections:
                    if scores[m] >= GT_THRESHOLD:
                        tracker_pool.add(Tracker(tmp_infos(m), self.t_Type))
                # unmatch trackers. e.g. one det with two trackers
                _t_pool = Tracker_pool()
                _t_pool.trackers = [tracker_pool.trackers[m] for m in unmatched_trackers]
                _ted_box = [tracked_boxes[m] for m in unmatched_trackers]
                update_infos, only_tracked = self.track_only_update(_t_pool, tracked_boxes=_ted_box)
                infos.extend(update_infos)
                for i, m in enumerate(unmatched_trackers):
                    tracker_pool.trackers[m] = _t_pool.trackers[i]
                only_tracked = 0
        
        ## Thrid: If there is no detection, update all tracker status and show tracker prediction
        else:
            infos, only_tracked = self.track_only_update(self.tracker_pool, infos=self.infos)

        # NMS
        keep = nms([info[:4]+[info[5]] for info in infos])
        delete = nms_2([info[:-1] for info in infos])
        for i in range(len(infos)):
            if i not in keep or i in delete:
                infos[i][:4] = [0]*4
        key_frames = tracker_pool.update_all()
        if len(key_frames) != 0:
            key_frames.append(get_id(self.image))
        return infos, tracker_pool, self.tracked_time, key_frames, only_tracked

    def track_only_update(self, tracker_pool, infos=[], tracked_boxes=[]):
        """
        This function is used to deal with 'no detection but tracked' situation

        Args:
            infos ([type]): [description]
            tracker_pool ([type]): [description]
            tracked_box: tracker prediction

        Returns: updated infos
        """
        frame, GT_THRESHOLD = self.frame, self.gt_threshold
        only_tracked = 0  # for test
        if tracked_boxes == []:
            tracked_boxes, self.tracked_time = get_track_bboxes(frame, tracker_pool)
        if len(tracked_boxes) == 1 and set(tracked_boxes[0]) == {0}:
            pass
        elif len(tracked_boxes) != 0:
            track_only_filter = []
            # calculate IoU to filter some unaccurate tracked bboxes
            tracker_bboxes = [t.bbox for t in tracker_pool.trackers]
            t_frame = [t.frame for t in tracker_pool.trackers]
            ious = iou_batch(tracked_boxes, tracker_bboxes)
            # calculate cosine similiarity
            dets = [get_roi(box, f) for box, f in zip(tracker_bboxes, t_frame)]
            trs = [get_roi(box, frame) for box in tracked_boxes]
            cosine_sim = cosine_distance(dets, trs)
            for i in range(len(tracked_boxes)):
                if cosine_sim[i] > 0.1 or ious[i][i] < self.tracked_threshold*2:
                    track_only_filter.append(i)
                    tracker_pool.update(i, state=-1)
                else:
                    tracker_pool.update(i, state=1)

            infos = generate_tracked_info(tracked_boxes, tracker_pool, GT_THRESHOLD)
            smoothed_boxes = []
            # get smoothed boxes
            for i, info in enumerate(infos):
                if i not in track_only_filter:
                    smoothed_b = tracker_pool.trackers[i].get_smoothed_cors(5, extra_box=[info[:4], info[4]])
                    smoothed_boxes.append(smoothed_b)
                    infos[i][:4] = smoothed_b
                else:
                    infos[i][:4] = [0]*4
            ## TODO: try to use image features cosine similarity instead of image (if need)

            # count tracked_only
            if len(track_only_filter) < len(tracked_boxes):
                only_tracked += 1
        return infos, only_tracked

