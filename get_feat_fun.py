import numpy as np
import operator
from mahotas.features import zernike_moments as zernike
from skimage.morphology import thin
from skimage.measure import label
from skimage.measure import regionprops
def get_features(x,mode,flat=False):
    """Get desired features from input dataset. mode is a string:'digits', 
    'zernike'. If flat is True, digits are returned in a stacked array, which 
    works with logreg, SVM, etc.. Otherwise, digits are returned as a list of 3
    seperated digits."""
    X = np.zeros(x.shape)
    for i,I in enumerate(x):
        X[i,:,:] = I > 220
    X_thin = np.zeros(x.shape)
    for i,I in enumerate(X):
        X_thin[i,:,:] = thin(I,8)
    X_lab = np.zeros(x1.shape)
    for i,I in enumerate(X_thin):
        X_lab[i,:,:] = label(I,background=0)
    X_prop = []
    for I in X_lab:
        X_prop.append(regionprops(np.uint64(I)))
    if mode == 'digits':
        X_sep = []
        for I in X_prop:
            areas = {}
            for i, lbl in enumerate(I):
                areas[i] = lbl.area
            sorted_areas = sorted(areas.items(), key=operator.itemgetter(1), reverse=True)[:3]
            img = []
            for i in sorted_areas:
                img.append(I[i[0]].image)
            if len(img) == 2:
                img.append(img[0])
            if len(img) == 1:
                img.extend([img[0], img[0]])
            X_sep.append(img)
        X_pad = []
        if flat:
            for n in X_sep:
                l = []
                for digit in n:
                    p = np.zeros([64,64], dtype=bool)
                    p[0:digit.shape[0], 0:digit.shape[1]] = digit
                    l.append(p.flatten())
                X_pad.append(np.hstack([l[0],l[1],l[2]]))
            return X_pad
        elif not flat:
            for n in X_sep:
                l = []
                for digit in n:
                    p = np.zeros([64,64], dtype=bool)
                    p[0:digit.shape[0], 0:digit.shape[1]] = digit
                    l.append(p)
## Uncomment to output single image with digits stacked vertically
#                X_pad.append(np.vstack([l[0],l[1],l[2]]))
##
                X_pad.append(l)
            return X_pad
    elif mode == 'zernike':
        X_sep = []
        for I in X_prop:
            areas = {}
            for i, lbl in enumerate(I):
                areas[i] = lbl.area
            sorted_areas = sorted(areas.items(), key=operator.itemgetter(1), reverse=True)[:3]
            img = []
            for i in sorted_areas:
                img.append(zernike(I[i[0]].image,30))
            if len(img) == 2:
                img.append(img[0])
            if len(img) == 1:
                img.extend([img[0], img[0]])
            X_sep.append(np.hstack([img[0],img[1],img[2]])) 
        return X_sep