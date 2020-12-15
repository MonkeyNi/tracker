import cv2
from utils import createTrackerByName, get_track_bboxes, generate_tracked_info
from utils import iou_batch, associate_detections_to_trackers, nms, nms_2
import numpy as np


class Tracker():
    def __init__(self, frame, infos, trackerType, age=2):
        """
        Initialize a tracker with frame and detection information.
        Args:
                frame ([image_name, np.array]): the frame name and value.
                infos ([list]): [boxes, scores, cats].
                trackerType: algorithm used to track
                age (int, optional): to record the tracker status. Defaults to 2.
        """
        self.frame_name = frame[0]
        self.frame = frame[1]
        self.bbox = infos[0]
        self.score = infos[1]
        self.cat = infos[2]
        self.trackerType = trackerType
        self.age = age
        tracker = cv2.MultiTracker_create()
        tracker.add(createTrackerByName(self.trackerType), self.frame, tuple(self.bbox))
        self.tracker = tracker
        self.tracked_pool = [infos[:-1]]
        # key frame
        self.key_frame = [frame, infos]
        self.key_f_score = self.key_frame[1][-1]
    
    def get_smoothed_cors(self, n, extra_box=False):
        """
        Average previous 5 dets result as showed bounding box
        """
        tracked_pool = self.tracked_pool
        if extra_box:
            tracked_pool.append(extra_box)
        avg_bboxes = np.array([b for b, _ in tracked_pool[-n:]])
        return list(np.average(avg_bboxes, axis=0).astype(np.int64))

    def update_key_frame(self, frame, infos):
        if infos[1] > self.key_f_score:
            self.key_frame = [frame, infos]
            self.key_f_score = infos[1]


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
        key_frames = [t.key_frame for t in key_trackers]
        
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
    
    def update_match(self, i, infos, frame, max_length=5):
        """
        Args:
            infos ([[]]]): [[box], score, catid]
        """
        info = infos[:-1]
        if info not in self.trackers[i].tracked_pool:
            self.trackers[i].tracked_pool.append(info)
            self.trackers[i].update_key_frame(frame, infos)
            self.trackers[i].tracked_pool = self.trackers[i].tracked_pool[-max_length:]


class Tracking():
    def __init__(self, 
                image, 
                frame, 
                infos, 
                tracker_pool, 
                tracked_threshold=0.3, 
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
        self.TrackerType = TrackerType

    def update(self):
        infos, tracker_pool, frame = self.infos, self.tracker_pool, self.frame
        GT_THRESHOLD = self.gt_threshold
        if not infos[0][:4] == [0]*4:
            bboxes, cats, scores = [i[:4] for i in infos], [i[4] for i in infos], [i[5] for i in infos]
            ## First: if track is null, just add frame and bbox to tracker
            if tracker_pool.empty():
                for i, box in enumerate(bboxes):
                    score, catid = float(infos[i][-2]), infos[i][-3]
                    if score >= GT_THRESHOLD:
                        tracker_pool.add(Tracker([self.image, frame], [box, score, catid], self.TrackerType))
            ## Second: track frame, update tracker pool
            else:
                tracked_boxes, self.tracked_time = get_track_bboxes(frame, tracker_pool)
                # match_2 has higher iou threshold and is used to update lesion tracker
                matched, matched_2, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(
                    bboxes, tracked_boxes, iou_threshold=self.tracked_threshold
                )
                
                def tmp_infos(ind):
                    return [bboxes[ind], scores[ind], cats[ind]]

                # update detection score
                for m in matched:
                    infos[m[0]][-2] = min((infos[m[0]][-2] + GT_THRESHOLD + 0.01), 0.99)
                    infos[m[0]][-1] = 1
                    tracker_pool.update(m[1], state=1)
                    # compute smooth cors
                    tracker_pool.update_match(m[1], tmp_infos(m[0]), [self.image, frame])
                    smoothed_cors = tracker_pool.trackers[m[1]].get_smoothed_cors(5)
                    infos[m[0]][:4] = smoothed_cors
                # update trackers
                for m in matched_2:
                    if scores[m[0]] >= GT_THRESHOLD:
                        updated_tracker = Tracker([self.image, frame], tmp_infos(m[0]), self.TrackerType)
                        tracker_pool.update_match(m[1], tmp_infos(m[0]), [self.image, frame])
                        updated_tracker.tracked_pool = tracker_pool.trackers[m[1]].tracked_pool
                        # update key frame
                        if updated_tracker.key_f_score < tracker_pool.trackers[m[1]].key_f_score:
                            updated_tracker.key_frame = tracker_pool.trackers[m[1]].key_frame
                        tracker_pool.trackers[m[1]] = updated_tracker
                for m in unmatched_detections:
                    if scores[m] >= GT_THRESHOLD:
                        tracker_pool.add(Tracker([self.image, frame], tmp_infos(m), self.TrackerType))
                for m in unmatched_trackers:
                    tracker_pool.update(m, state=-1)
        
        ## Thrid: If there is no detection, update all tracker status and show tracker result
        else:
            tracker_pool.update_miss()
            tracked_boxes, self.tracked_time = get_track_bboxes(frame, tracker_pool)
            if len(tracked_boxes) != 0:
                infos = generate_tracked_info(tracked_boxes, tracker_pool, GT_THRESHOLD)
                smoothed_boxes = []
                # get smoothed boxes
                for i, info in enumerate(infos):
                    smoothed_b = tracker_pool.trackers[i].get_smoothed_cors(5, extra_box=[info[:4], info[4]])
                    smoothed_boxes.append(smoothed_b)
                    infos[i][:4] = smoothed_b
                # use iou to filter some unaccurate tracked bboxes
                ## TODO: try to use some features
                tracker_bboxes = [t.bbox for t in tracker_pool.trackers]
                ious = iou_batch(smoothed_boxes, tracker_bboxes)
                for i in range(len(smoothed_boxes)):
                    if ious[i][i] < self.tracked_threshold*2:
                        infos[i][:4] = [0]*4
        
        # NMS
        keep = nms([info[:4]+[info[5]] for info in infos])
        delete = nms_2([info[:-1] for info in infos])
        for i in range(len(infos)):
            if i not in keep or i in delete:
                infos[i][:4] = [0]*4
        key_frames = tracker_pool.update_all()
        return infos, tracker_pool, self.tracked_time, key_frames
