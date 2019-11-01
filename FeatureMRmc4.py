from skimage.segmentation import slic
import cv2
import numpy as np
from scipy.sparse import coo_matrix, dia_matrix, eye
from scipy.sparse.linalg import inv, spsolve
from myfunc import make_graph, lbmap_from_sp
import functools

theta = 10
alpha = 0.99


def FeatureMR2(Feature):
    img = Feature
    # img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = img[:, :, ::-1]

    # superpixel label map
    numSegments = 1000
    # It is essential for SLIC to work in Lab color space to obtain good results
    sp_label = slic(img, n_segments=numSegments, compactness=0.01, multichannel=True)
    # sp_label_temp=np.reshape(sp_label,[sp_label.shape[0]*sp_label.shape[1]])
    # sp_label=Lables
    sp_num = sp_label.max() + 1

    # superpixel vector holds Lab value
    sp_img = []
    for i in range(sp_num):
        # data=img_lab[sp_label == i, :]
        # print(data)
        sp_img.append(img[sp_label == i, :].mean(0, keepdims=False))
        _y, _x = np.where(sp_label == i)
    sp_img = np.array(sp_img)

    # affinity matrix
    edges = make_graph(sp_label)
    # edges = np.concatenate((np.stack((np.arange(sp_num), np.arange(sp_num)), 1), edges), 0)

    weight = np.sqrt(np.sum((sp_img[edges[:, 0]] - sp_img[edges[:, 1]]) ** 2, 1))
    weight = (weight - np.min(weight, axis=0, keepdims=True)) \
             / (np.max(weight, axis=0, keepdims=True) - np.min(weight, axis=0, keepdims=True))
    weight = np.exp(-weight * theta)

    W = coo_matrix((
        np.concatenate((weight, weight)),
        (
            np.concatenate((edges[:, 0], edges[:, 1]), 0),
            np.concatenate((edges[:, 1], edges[:, 0]), 0)
        )))
    dd = W.sum(0)
    D = dia_matrix((dd, 0), (sp_num, sp_num)).tocsc()

    optAff = spsolve(D - alpha * W, eye(sp_num).tocsc())
    optAff -= dia_matrix((optAff.diagonal(), 0), (sp_num, sp_num))

    """stage 1"""
    top_bd = np.unique(sp_label[0, :])
    left_bd = np.unique(sp_label[:, 0])
    bottom_bd = np.unique(sp_label[-1, :])
    right_bd = np.unique(sp_label[:, -1])
    bds = [top_bd, left_bd, bottom_bd, right_bd]
    bsal = []
    for bd in bds:
        seed = np.zeros(sp_num)
        seed[bd] = 1
        _bsal = optAff.dot(seed)
        _bsal = (_bsal - _bsal.min()) / (_bsal.max() - _bsal.min())
        bsal.append(1 - _bsal)
    bsal = functools.reduce(lambda x, y: x * y, bsal)
    bsal = (bsal - bsal.min()) / (bsal.max() - bsal.min())

    r_lbmap1 = lbmap_from_sp(bsal, sp_label)

    """stage 2"""
    seed = np.zeros(sp_num)
    seed[bsal > bsal.mean()] = 1

    fsal = optAff.dot(seed)
    fsal = (fsal - fsal.min()) / (fsal.max() - fsal.min())

    r_lbmap = lbmap_from_sp(fsal, sp_label)

    FinalSal = r_lbmap

    return FinalSal, sp_label, r_lbmap1
