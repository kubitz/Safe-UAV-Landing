import math
import cv2 as cv
import numpy as np
lzs=[
    {
        'confidence':0.5,
        'position': [500,500],
        'radius':100
    }
]

class bidict(dict):
    """Bi-directional dictionary for label Lookup. 
    Implementation by Basj 

    Args:
        dict (dictonary): dict to be made bi-directional
    """
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value,[]).append(key) 

    def __setitem__(self, key, value):
        if key in self:


            self.inverse[self[key]].remove(key) 
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value,[]).append(key)        

    def __delitem__(self, key):
        self.inverse.setdefault(self[key],[]).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]: 
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)

labelsAero = {
    'background':0,
    'person':1,
    'bike':2,
    'car':3,
    'drone':4,
    'boat':5,
    'animal':6,
    'obstacle':7,
    'construction':8,
    'vegetation':9,
    'road':10,
    'sky':11
}

labelsGraz = {
    'unlabeled':0,
    'pavedArea':1,
    'dirt':2,
    'grass':3,
    'dirt':4,
    'water':5,
    'rocks':6,
    'pool':7,
    'lowVegetation':8,
    'roof':9,
    'wall':10,
    'window':11,
    'door':12,
    'fence':13,
    'fencePole':14,
    'person':15,
    'animal':16,
    'car':17,
    'bike':18,
    'tree':19,
    'baldTree':20,
    'arMarker':21,
    'obstacle':22,
    'conflicting':23
}

notSafe=['water','fence','fencePole','animal','car','bicycle','tree','baldTree'
        'obstacle','pool','wall','door','roof','drone','construction','boat']


biLabelsAero=bidict(labelsAero)
biLabelsGraz=bidict(labelsGraz)

labels={
    'aeroscapes':biLabelsAero,
    'graz':biLabelsGraz
}


def decodeRiskIds(riskIds,lb):
    """Checks whether any of the risk ids is considered unsafe

    Args:
        riskIds (list of labels - integers): list of labels detected in the potential lz
        lb (bidict): bidirectional dictionary with labels and ids

    Returns:
        bool: if the zone is safe, True otherwise False
        list: if the zone is unsafe, list of reasons why, otherwise empty list
    """
    isSafe=True
    reasons=[]
    for riskId in riskIds:
        risk=lb.inverse[riskId][0]
        if risk in notSafe:
            isSafe=False
            reasons.append(risk)
    return isSafe,reasons

def getLzCrop(riskGt,lz):
    posLz=lz.get('position')
    radiusLz=lz.get('radius')
    mask = np.zeros_like(riskGt)
    mask = cv.circle(mask, (posLz[0],posLz[1]), radiusLz, (255,255,255), -1)
    crop = cv.bitwise_and(riskGt, mask)
    return crop


dir_seg= r"/home/kubitz/Documents/fyp/Safe-UAV-Landing/data/test/images/040003_017.jpg"
dir_seg= r"/home/kubitz/Documents/fyp/Safe-UAV-Landing/data/test/segmentation/040003_017.png"
lb=labels.get("aeroscapes")
print(lb.inverse[3])

seg_img = cv.imread(dir_seg)

image,_,_ = cv.split(seg_img)
risk_gt = image.astype('uint8')
crop=getLzCrop(risk_gt,lzs[0])
cv.imshow("best landing zones",cv.applyColorMap(crop, cv.COLORMAP_JET))
cv.waitKey(0)

riskIds=np.unique(crop)
print(riskIds)
isSafe, reasons = decodeRiskIds(riskIds,lb)
print('is Safe: ', isSafe)
print('Reasons', reasons )