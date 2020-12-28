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
        if state >= 1:
            self.trackers[i].age = 2  # reinitialize age
        else:
            self.trackers[i].age -= 1

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
                 tracked_threshold=0.25,
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
        self.bboxes, self.scores, self.cats = [], [], []

    def update(self):
        infos, tracker_pool, frame = self.infos, self.tracker_pool, self.frame
        # tracked_info = [[t.bbox, t.age] for t in tracker_pool.trackers]  # for test
        # # print(tracked_info)
        GT_THRESHOLD = self.gt_threshold
        only_tracked = 0  # for test
        if not infos[0][:4] == [0]*4:
            self.bboxes = [i[:4] for i in infos]
            self.cats = [i[4] for i in infos]
            self.scores = [i[5] for i in infos]
            ## 1 if track is null, just add frame and bbox to tracker
            if tracker_pool.empty():
                for i, box in enumerate(self.bboxes):
                    score, catid = float(infos[i][-2]), infos[i][-3]
                    if score >= GT_THRESHOLD:
                        tracker_pool.add(Tracker([[self.image, frame], [box, score, catid]], self.t_Type))
            ## 2 compare det results first (dets and box in tracker pool), which can save some track time
            else:
                tracker_bboxes = [t.bbox for t in tracker_pool.trackers]
                matched_dets, unmatched_dets, unmatched_trs = associate_detections_to_trackers(
                        self.bboxes,
                        tracker_bboxes,
                        iou_threshold=0.6
                    )
                # 2.1 for those matched dets, no need to do track anymore
                infos, tracker_pool = self.match(matched_dets, infos, tracker_pool)
                # 2.2 no tracker left, only det left
                if len(unmatched_dets) != 0 and len(unmatched_trs) == 0:
                    tracker_pool = self.unmatched_dets(unmatched_dets, tracker_pool)
                # 2.3 there are tracker left
                elif len(unmatched_trs) != 0:
                    tmp_t_pool = Tracker_pool()
                    tmp_t_pool.trackers = [tracker_pool.trackers[m] for m in unmatched_trs]
                    tracked_boxes, self.tracked_time = get_track_bboxes(frame, tmp_t_pool)
                    # 2.3.1 only tracker left and no det left
                    if len(unmatched_dets) == 0:
                        update_infos, _ = self.track_only_update(tmp_t_pool, tracked_boxes=tracked_boxes)
                        for i, m in enumerate(unmatched_trs):
                            tracker_pool.trackers[m] = tmp_t_pool.trackers[i]
                        infos.extend(update_infos)
                    # 2.3.2 there are both det and tracker left (normal situation)
                    else:
                        tmp_box = [self.bboxes[m] for m in unmatched_dets]
                        tmp_infos = [infos[m] for m in unmatched_dets]
                        matched_ds, unmatched_ds, unmatched_ts = associate_detections_to_trackers(
                            tmp_box,
                            tracked_boxes,
                            iou_threshold=self.tracked_threshold
                        )
                        tmp_infos, tmp_t_pool = self.match(matched_ds, tmp_infos, tmp_t_pool)
                        tmp_t_pool = self.unmatched_dets(unmatched_ds, tmp_t_pool)
                        tmp_update_infos, tmp_t_pool = self.unmatched_ts(unmatched_ts, tmp_t_pool, tracked_boxes)
                        tmp_infos.extend(tmp_update_infos)
                        for i, m in enumerate(unmatched_dets):
                            infos[m] = tmp_infos[i]
                        for i, m in enumerate(unmatched_trs):
                            tracker_pool.trackers[m] = tmp_t_pool.trackers[i]

        ## 3 If there is no detection, update all tracker status and show tracker prediction
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
        only_tracked = 0  # for test
        if tracked_boxes == []:
            tracked_boxes, self.tracked_time = get_track_bboxes(self.frame, tracker_pool)
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
            trs = [get_roi(box, self.frame) for box in tracked_boxes]
            cosine_sim = cosine_distance(dets, trs)
            for i in range(len(tracked_boxes)):
                if cosine_sim[i] > 0.1 or ious[i][i] < self.tracked_threshold:
                    track_only_filter.append(i)
                    tracker_pool.update(i, state=-1)
                else:
                    tracker_pool.update(i, state=1)

            infos = generate_tracked_info(tracked_boxes, tracker_pool, self.gt_threshold)
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
                only_tracked = 1
        return infos, only_tracked

    def match(self, matched, infos, tracker_pool):
        """
        Tracker prediction match dets
        """
        for m in matched:
            infos[m[0]][-2] = min((infos[m[0]][-2] + self.gt_threshold + 0.01), 0.99)
            infos[m[0]][-1] = 1
            tracker_pool.update(m[1], state=1)
            # compute smooth cors
            tracker_pool.update_match(m[1], self.tmp_infos(m[0], ))
            smoothed_cors = tracker_pool.trackers[m[1]].get_smoothed_cors(5)
            infos[m[0]][:4] = smoothed_cors
            # update trackers
            if self.scores[m[0]] >= self.gt_threshold:
                tracker_pool.update_poolTracker(m, self.tmp_infos(m[0]), self.t_Type)
        return infos, tracker_pool

    def unmatched_dets(self, unmatched_detections, tracker_pool):
        """
        Deal with unmatched detection results
        """
        for m in unmatched_detections:
            if self.scores[m] >= self.gt_threshold:
                tracker_pool.add(Tracker(self.tmp_infos(m), self.t_Type))
        return tracker_pool

    def unmatched_ts(self, unmatched_trackers, tracker_pool, tracked_boxes):
        """
        Deal with unmatched trackers
        """
        _t_pool = Tracker_pool()
        _t_pool.trackers = [tracker_pool.trackers[m] for m in unmatched_trackers]
        _ted_box = [tracked_boxes[m] for m in unmatched_trackers]
        update_infos, _ = self.track_only_update(_t_pool, tracked_boxes=_ted_box)
        for i, m in enumerate(unmatched_trackers):
            tracker_pool.trackers[m] = _t_pool.trackers[i]
        return update_infos, tracker_pool

    def tmp_infos(self, ind):
        """
        Help function
        """
        return [[self.image, self.frame], [self.bboxes[ind], self.scores[ind], self.cats[ind]]]
