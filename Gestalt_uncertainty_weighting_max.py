"""

Gestalt Theory Uncertainty Weigthing Method between two Given Features.
This code is implemented by Guanqun Ding, please cite the paper of my tutor Yuming Fang when using the code:

Yuming Fang, Zhou Wang, Weisi Lin, and Zhijun Fang, Video Saliency Incorporating Spatiotemporal Cues and Uncertainty Weighting. IEEE Transactions on Image Processing 23(9), 3910-3921, 2014.

Usage:

weighted_smap=get_Uncertainty_Weighting_map(feature1, feature2)

feature1 and feature2 are the two feature maps need to combined.

"""

import warnings
import math
from operator import mul
from functools import reduce
import torch.nn.functional as F
import numpy as np
import cv2

from skimage.segmentation import slic

import torch
from torch._C import _infer_size, _add_docstr


def Tensor_padding2d(input_tensor, filter_size=5, stride_size=1):
    H = input_tensor.shape[0]
    W = input_tensor.shape[1]

    Ph = int(np.max((H - 1) * stride_size + filter_size - H, 0))
    Pw = int(np.max((W - 1) * stride_size + filter_size - W, 0))

    Pt = int(np.floor(Ph / 2))
    Pl = int(np.floor(Pw / 2))
    Pb = int((Ph - Pt))
    Pr = int((Pw - Pl))

    p2d = (Pl, Pr, Pt, Pb)  # pad last dim by (1, 1) and 2nd to last by (2, 2)
    # p2d = (2, 2, 2, 2)
    output = F.pad(input_tensor, p2d, 'constant', 0)

    return output


def featureNorm(features):
    feat_norm = (features - features.min()) / (features.max() - features.min())
    return feat_norm


def GPL(input_tensor, filter_size=7, stride_size=1, padding=False):
    if padding == True:
        input_tensor = Tensor_padding2d(input_tensor=input_tensor, filter_size=filter_size, stride_size=stride_size)

    labels = slic(featureNorm(input_tensor), n_segments=3, compactness=3)
    labels = labels.reshape(input_tensor.shape[0] * input_tensor.shape[1])
    u_labels = np.unique(labels)
    l_inds = []
    for i in range(len(u_labels)):
        l_inds.append(np.where(labels == u_labels[i])[0])

    input_numpy = input_tensor.data.cpu().numpy()
    input_numpy1 = np.reshape(input_numpy, [input_tensor.shape[0] * input_tensor.shape[1]])

    input_numpy_mean = []
    for j in range(len(u_labels)):
        input_numpy_mean.append(np.mean(input_numpy1[l_inds[j]]))

    max_center_loca = (np.argmax(input_numpy_mean))

    return input_tensor


def get_size(feat):
    return feat.shape[0], feat.shape[1]


def get_row_col_array(row_size, col_size):
    row_array = np.zeros([row_size, col_size])
    col_array = np.zeros([row_size, col_size])
    for i in range(row_size):
        row_array[i, :] = np.ones(col_size) * (i + 1)
    for j in range(col_size):
        col_array[:, j] = np.ones(col_size) * (j + 1)
    return row_array, col_array


def get_expected_row_col(feature, row_array, col_array):
    expected_row = np.round((np.mean(feature * row_array)) / (np.mean(feature)))
    expected_col = np.round((np.mean(feature * col_array)) / (np.mean(feature)))
    return int(expected_row), int(expected_col)


def get_max_val_loc(feature):
    max_val_loc = np.where(feature == np.max(feature))
    expected_row = max_val_loc[0]
    expected_col = max_val_loc[1]

    return int(expected_row[0]), int(expected_col[0])


def get_closeness_uncertanty(feature, isMaxCenter=False):
    row, col = get_size(feature)
    row_array, col_array = get_row_col_array(row_size=row, col_size=col)

    if isMaxCenter == True:
        expected_row, expected_col = get_max_val_loc(feature)
    else:
        expected_row, expected_col = get_expected_row_col(feature, row_array, col_array)

    dist_row = np.power((row_array - expected_row), 2)
    dist_col = np.power((col_array - expected_col), 2)

    spatial_dist = np.round(np.sqrt(dist_row + dist_col))

    closeness_prob = np.exp(-(np.power(spatial_dist, 2)) / (np.power(92, 2)))

    uncertanty_dis = -((closeness_prob + 0.001) * (np.log2(closeness_prob + 0.001) + ((1 - closeness_prob) + 0.001) * (
        np.log2((1 - closeness_prob) + 0.001))))
    return uncertanty_dis


def get_sub_pixels(feature, start_row, end_row, start_col, end_col):
    row, col = get_size(feature)

    start_row = start_row
    end_row = row + end_row
    start_col = start_col
    end_col = col + end_col

    new_row = abs(end_row - start_row)
    new_col = abs(end_col - start_col)

    sub_pixels = np.zeros([new_row, new_col])
    for i in range(new_row):
        for j in range(new_col):
            sub_pixels[i, j] = feature[start_row + i, start_col + j]
    return sub_pixels


def get_continuty_uncertanty(feature):
    row, col = get_size(feature)
    used_array = np.zeros([row - 2, col - 2, 8])
    used_connected_num = np.zeros([row - 2, col - 2])
    connected_num = np.zeros([row, col])

    # get 8 neighboring pixle set
    used_array[:, :, 0] = get_sub_pixels(feature, 0, -2, 0, -2)
    used_array[:, :, 1] = get_sub_pixels(feature, 0, -2, 1, -1)
    used_array[:, :, 2] = get_sub_pixels(feature, 0, -2, 2, 0)
    used_array[:, :, 3] = get_sub_pixels(feature, 1, -1, 0, -2)
    used_array[:, :, 4] = get_sub_pixels(feature, 1, -1, 2, 0)
    used_array[:, :, 5] = get_sub_pixels(feature, 2, 0, 0, -2)
    used_array[:, :, 6] = get_sub_pixels(feature, 2, 0, 1, -1)
    used_array[:, :, 7] = get_sub_pixels(feature, 2, 0, 2, 0)

    used_connected_num[:, :] = np.round(np.mean(np.array(used_array), 2))
    connected_num[1:-1, 1:-1] = used_connected_num
    connectedness_prob = np.exp(- np.power((np.array(connected_num) - 8), 2) / (np.power(4, 2)))

    uncertanty_connect = -((connectedness_prob + 0.001) * (
                np.log2(connectedness_prob + 0.001) + ((1 - connectedness_prob) + 0.001) * (
            np.log2((1 - connectedness_prob) + 0.001))))
    return uncertanty_connect


def get_Uncertainty_Weighting_map(feature1, feature2):
    ROW = feature1.shape[0]
    COL = feature2.shape[1]

    feature1 = cv2.resize(feature1, (int((ROW + COL) / 2), int((ROW + COL) / 2)))
    feature2 = cv2.resize(feature2, (int((ROW + COL) / 2), int((ROW + COL) / 2)))

    uncertanty_dis1 = get_closeness_uncertanty(feature1, isMaxCenter=True)
    uncertanty_connect1 = get_continuty_uncertanty(feature1)
    uncertanty1 = uncertanty_dis1 + uncertanty_connect1

    uncertanty_dis2 = get_closeness_uncertanty(feature2, isMaxCenter=True)
    uncertanty_connect2 = get_continuty_uncertanty(feature2)
    uncertanty2 = uncertanty_dis2 + uncertanty_connect2

    weighted_smap = (uncertanty2 * feature1 + uncertanty1 * feature2) / (uncertanty1 + uncertanty2)

    weighted_smap = cv2.resize(weighted_smap, (COL, ROW))

    return weighted_smap

# feature1 = cv2.imread("static_smap.jpg", cv2.IMREAD_GRAYSCALE)
# feature2 = cv2.imread("motion_smap.jpg", cv2.IMREAD_GRAYSCALE)
# weighted_smap=get_Uncertainty_Weighting_map(feature1, feature2)
#
# cv2.imshow("weighted_smap",weighted_smap.astype(np.uint8))
# cv2.imshow("feature1", feature1.astype(np.uint8))
# cv2.imshow("feature2", feature2.astype(np.uint8))
# cv2.waitKey(0)
