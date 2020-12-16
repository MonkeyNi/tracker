## Tracker for object detection

### Test example
```
python track.py --video test/test2.mp4 --detection_result test/results2_0.1.txt

Detection result should has format:
    frame_name, x1, y1, x2, y2, categort_id, score
```
There will be an video, tracked result txt fie, and key frames in output. Output will be saved at the same folder as 'detection result'.

### Key points:<br>
1. The workflow is based on SORT;<br>
2. Tracking algorithm is from openCV API, default is 'medianflow';<br>
3. Data association algorithm is Hungarian algorithm (which can be further replaced by more advanced algorithm, e.g. KM or feature releated algorithm);<br>

### Dependency
1. OpenCV
2. Numpy

### 20201215 Update
1. NMS
2. Key frame extraction


### TODO
1. How to use class information;