## Tracker for object detection

### Test example
```
python track.py --video path_to_test_video --detection_result path_to_detection_result

Detection result should has format:
    frame_name, x1, y1, x2, y2, categort_id, score
```
There will be an video and tracked result in output. Output will be saved at the same folder as 'detection result'.

### Key points:<br>
1. The workflow is based on SORT;<br>
2. Tracking algorithm is from openCV API, default is 'medianflow';<br>
3. Data association algorithm is Hungarian algorithm (which can be further replaced by more advanced algorithm, e.g. KM or feature releated algorithm);<br>

### Dependency
1. OpenCV
2. Numpy

### TODO
1. Get key frames;
2. How to use class information;