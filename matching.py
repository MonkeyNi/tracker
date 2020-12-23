import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import cv2


def norm_image(image):
    """
    Convert roi to embedding

    Args:
        image (np.array): ROI
    """
    ## Gray match
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # img = img.flatten()
    # return img

    # All channel match
    img_embedding = []
    for i in range(3):
        img_embedding.append(image[:,:,i].flatten())
    return img_embedding


def cosine_distance(dets, trs, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.
    Parameters
    ----------
    dets : array_like
        An list of N GT ROI
    trs : array_like
        An lsit of N tracked ROI
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.
    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
    """

    trs = [cv2.resize(b, a.shape[:2][::-1]) for a, b in zip(dets, trs)]
    dets = [norm_image(a) for a in dets]
    trs = [norm_image(a) for a in trs]
    res = []
    if not data_is_normalized:
        for a, b in zip(dets, trs):
            a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
            b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
            res.append(1. - (np.dot(a, b.T)).diagonal().mean())
    return res
    