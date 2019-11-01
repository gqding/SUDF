import argparse
import torch
import h5py
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import cv2
import numpy as np
import torch.nn.init
import os
import scipy.io
from FeatureMRmc4 import FeatureMR2
import timeit
from skimage.segmentation import slic
from Gestalt_uncertainty_weighting_max import get_Uncertainty_Weighting_map


def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def Load_data(File_path):
    Arrays = {}
    HS_files = os.listdir(File_path)

    num = 0
    for file in HS_files:
        input_img = os.path.join(File_path, file)
        print("Reading File[%d/%d]: %s" % (num, len(HS_files), input_img))
        f = h5py.File(input_img)
        for k, v in f.items():
            Arrays[file[:-4]] = np.array(v)
            # print(k)
            # print(v.shape)
        num = num + 1
    return Arrays


def featureNorm(features):
    feat_norm = (features - features.min()) / (features.max() - features.min())
    return feat_norm


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def reAssign_lables(srcLables):
    new_Lables = np.zeros([srcLables.shape[0] * srcLables.shape[1], 1], dtype=np.int64)
    resLables = np.reshape(srcLables, [srcLables.shape[0] * srcLables.shape[1]])
    u_labels = np.unique(resLables)

    for i in range(len(u_labels)):
        lable_Loc = np.where(resLables == u_labels[i])
        # print(u_labels[i])
        # print((srcLables == u_labels[i]))
        new_Lables[lable_Loc[:]] = i

    new_res_Lables = np.reshape(new_Lables, [srcLables.shape[0], srcLables.shape[1]])
    return new_res_Lables


def cluster_refinement(output_numpy, slic_lables, im_target):
    norm_features = featureNorm(output_numpy)
    labels = slic_lables
    labels = labels.reshape(norm_features.shape[0] * norm_features.shape[1])
    u_labels = np.unique(labels)
    l_inds = []
    for i in range(len(u_labels)):
        l_inds.append(np.where(labels == u_labels[i])[0])
    ## sp_label = slic(img, n_segments=numSegments, compactness=0.01, multichannel=True)

    for i in range(len(l_inds)):
        labels_per_sp = im_target[l_inds[i]]
        u_labels_per_sp = np.unique(labels_per_sp)
        hist = np.zeros(len(u_labels_per_sp))
        for j in range(len(hist)):
            hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
        im_target[l_inds[i]] = u_labels_per_sp[np.argmax(hist)]

    return im_target


use_cuda = torch.cuda.is_available()
nChannel = 64
maxIter = 100
minLabels = 3
lr = 0.01
nConv = 4  # dft=2
num_superpixels = 1000
compactness = 100
visualize = 1
step_loss = 0
data_path = './HS_images/'
save_path = './HS_Results/'
mkdir(save_path)

sal_result_path = os.path.join(save_path, 'sal_result')
mkdir(sal_result_path)
slic_lable_path = os.path.join(save_path, 'slic_lable')
mkdir(slic_lable_path)
batch_sal_result_path = os.path.join(save_path, 'batch_sal_result')
mkdir(batch_sal_result_path)
batch_bsal_result_path = os.path.join(save_path, 'batch_bsal_result')
mkdir(batch_bsal_result_path)
batch_weighted_sal_result_path = os.path.join(save_path, 'weighted_sal_result')
mkdir(batch_weighted_sal_result_path)
batch_RGB_result_path = os.path.join(save_path, 'batch_RGB_result')
mkdir(batch_RGB_result_path)
im_target_rgb_path = os.path.join(save_path, 'im_target_rgb')
mkdir(im_target_rgb_path)
im_max_Val_path = os.path.join(save_path, 'im_max_Val')
mkdir(im_max_Val_path)
im_target_path = os.path.join(save_path, 'im_target')
mkdir(im_target_path)


# CNN model
class MyNet(nn.Module):
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, nChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = []
        self.bn2 = []
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        for i in range(nConv - 1):
            self.conv2.append(nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(nChannel))

        self.conv3 = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(nChannel)
        self.UB1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv = nn.ConvTranspose2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)
        for i in range(nConv - 1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
            if i == 0:
                x = self.pool2(x)
        x = self.conv3(x)
        # x = F.relu(x)
        x = self.bn3(x)

        x = self.UB1(x)
        x = self.deconv(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.UB1(x)
        x = self.deconv(x)
        x = F.relu(x)
        x = self.bn3(x)

        return x


HS_files = os.listdir(data_path)

Changes = 'average the iteration results'
log_path = save_path + "TrainingLog.txt"
with open(log_path, "w") as Traininglog:
    Traininglog.write(" %s:%d, %s:%d, %s:%d \n" % ('nChannel', nChannel, 'maxIter', maxIter, 'nConv', nConv))
    Traininglog.write(" %s:%s \n" % ('Changes', Changes))

    Data_array = Load_data(File_path=data_path)

    for key, values in Data_array.items():
        file_name = key
        arrays = values

        raw_HS_data = np.transpose(np.array(arrays.astype('double') / 4095.0), (1, 2, 0))  # [1024,768,81]
        data = torch.from_numpy(np.expand_dims(np.array(arrays.astype('float32') / 4095.), axis=0))

        if use_cuda:
            data = data.cuda()
        data = Variable(data)
        # slic
        slic_lables_raw = slic(raw_HS_data, n_segments=num_superpixels, compactness=0.01, multichannel=True)
        # train
        model = MyNet(data.size(1))
        if use_cuda:
            model.cuda()
            for i in range(nConv - 1):
                model.conv2[i].cuda()
                model.bn2[i].cuda()
        model.train()

        loss_fn = torch.nn.CrossEntropyLoss()
        loss_fn2 = torch.nn.MSELoss()

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        label_colours = np.random.randint(255, size=(100, 3))

        # step_loss_all=np.zeros([maxIter])

        stop_loss_former = 1000
        sal_former = np.zeros([arrays.shape[1], arrays.shape[2]])
        stop_loss = 100
        thres = 1e-3
        thres2 = 1e-2
        thres3 = 0.025

        Traininglog.write("%s \n" % file_name)
        start_time = timeit.default_timer()
        new_sal_all = np.zeros([maxIter, arrays.shape[1], arrays.shape[2]])

        for batch_idx in range(maxIter):
            # forwarding
            optimizer.zero_grad()
            output = model(data)[0]  # [1,100,1024,768] -> [100,1024,768]
            output_numpy = output.permute(1, 2, 0).data.cpu().numpy()  # [1024,768,100]
            print('Iteration: %d,' % batch_idx, output.shape, model(data).shape)
            output = output.permute(1, 2, 0).contiguous().view(-1, nChannel)  # [1024,768,100] -> [1024*768, 100]
            ignore, target = output.max(1)  # [1024*768,100] -> refer to 2th dim -> [1024*768]
            # print(target.shape)

            im_target = target.data.cpu().numpy()  # lables
            im_ignore = ignore.data.cpu().numpy()  # max values
            nLabels = len(np.unique(im_target))
            # if visualize:
            im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
            im_target_rgb = im_target_rgb.reshape([arrays.shape[1], arrays.shape[2], 3]).astype(np.uint8)
            im_max_Val = im_ignore.reshape([arrays.shape[1], arrays.shape[2]]).astype(np.uint8)

            im_target_reshape = im_target.reshape([arrays.shape[1], arrays.shape[2]])
            new_labels = reAssign_lables(im_target_reshape)
            # sal=FeatureMR(Feature=featureNorm(output_numpy),Lables=new_labels)
            sal2, slic_lables_feat, bsal = FeatureMR2(Feature=featureNorm(output_numpy.astype(np.double)))
            new_sal = np.array(1 * sal2)  # only use slic lables for MR
            # print(bsal)
            weighted_smap = get_Uncertainty_Weighting_map(sal2, bsal)
            new_sal_all[batch_idx, :, :] = weighted_smap[:, :]
            # superpixel refinement
            # TODO: use Torch Variable instead of numpy for faster calculation

            # slic
            im_target = cluster_refinement(output_numpy=output_numpy, slic_lables=slic_lables_raw, im_target=im_target)

            # sal_reshape=new_sal.reshape([arrays['hypercube'].shape[1] * arrays['hypercube'].shape[2]])
            target = torch.from_numpy(im_target)
            # sal_torch=torch.from_numpy(sal_reshape).float()
            # target_mean=torch.from_numpy(target_mean).float()
            batch_sal_result_name = batch_sal_result_path + '/' + file_name + '_' + str(batch_idx) + '.png'
            cv2.imwrite(batch_sal_result_name, cv2.transpose(new_sal * 255.0))
            batch_RGB_result_name = batch_RGB_result_path + '/' + file_name + '_' + str(batch_idx) + '.png'
            cv2.imwrite(batch_RGB_result_name, cv2.transpose(im_target_rgb))

            batch_bsal_result_name = batch_bsal_result_path + '/' + file_name + '_' + str(batch_idx) + '.png'
            cv2.imwrite(batch_bsal_result_name, cv2.transpose(bsal * 255.0))

            batch_weighted_sal_result_name = batch_weighted_sal_result_path + '/' + file_name + '_' + str(
                batch_idx) + '.png'
            cv2.imwrite(batch_weighted_sal_result_name, cv2.transpose(weighted_smap * 255.0))

            if use_cuda:
                target = target.cuda()
            target = Variable(target)

            loss1 = loss_fn(output, target)
            final_loss = loss1
            use_sal_loss = False

            # stop conditions
            stop_loss = loss1.data.cpu().numpy()  # cluster loss here
            sal_distance = np.mean(np.abs(sal_former - new_sal))
            loss_distance = math.fabs(stop_loss - stop_loss_former)
            # print(sal_distance)
            stop_loss_former = stop_loss
            sal_former = new_sal

            final_loss.backward()
            optimizer.step()

            # Traininglog.write(" %d / %d : %d, %s \n" % (batch_idx, maxIter, nLabels, str(final_loss.data.cpu().numpy())))
            print(batch_idx, '/', maxIter, ':', nLabels, final_loss.data)
            if nLabels <= minLabels:
                print("nLabels", nLabels, "reached minLabels", minLabels, ".")
                break

            # save output image
            if not visualize:
                output = model(data)[0]
                output = output.permute(1, 2, 0).contiguous().view(-1, nChannel)
                ignore, target = torch.max(output, 1)
                im_target = target.data.cpu().numpy()
                im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
                im_target_rgb = im_target_rgb.reshape([arrays.shape[1], arrays.shape[2], 3]).astype(np.uint8)

        ave_sal = np.mean(new_sal_all, axis=0)

        save_im_target_rgb_name = im_target_rgb_path + '/' + file_name + '.png'
        save_maxVal_name = im_max_Val_path + '/' + file_name + '.png'
        save_im_target_name = im_target_path + '/' + file_name + '.mat'
        save_sal_result_name = sal_result_path + '/' + file_name + '.jpg'
        slic_lable_name = slic_lable_path + '/' + file_name + '.mat'

        cv2.imwrite(save_sal_result_name, cv2.transpose(ave_sal * 255.0))
        cv2.imwrite(save_im_target_rgb_name, cv2.transpose(im_target_rgb))
        cv2.imwrite(save_maxVal_name, cv2.transpose(im_max_Val))
        scipy.io.savemat(save_im_target_name, {'data': cv2.transpose(im_target)})
        scipy.io.savemat(slic_lable_name, {'lable': cv2.transpose(slic_lables_raw)})

        stop_time = timeit.default_timer()
        time_cost = stop_time - start_time
        mean_time = time_cost / maxIter
        Traininglog.write(" %s: %d, %s: %d\n" % (time_cost, time_cost, mean_time, mean_time))
        # print(time_cost)

print('Iteration finished!')
