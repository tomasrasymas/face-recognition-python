from skimage import feature
import numpy as np


class FaceFeatures:
    def __init__(self):
        pass

    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, 24, 8, method="default")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 24 + 3), range=(0, 24 + 2))

        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        hog = feature.hog(image)

        return np.concatenate((hog, hist))