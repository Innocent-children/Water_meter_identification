from datetime import datetime
import os
import pymysql
import shutil

from flask import Blueprint, Flask, jsonify, url_for, request, redirect, render_template, Response, \
    session, g, \
    flash
import config
from werkzeug.utils import secure_filename
####################################################################################################
import os
import warnings
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import cv2
import math
import pyclipper
import imgaug
import imgaug.augmenters as iaa

from PIL import Image
from torchvision import transforms
from shapely.geometry import Polygon
from collections import OrderedDict

from tqdm import tqdm

from blueprints.forms import DelForm
from decorators import login_required


class DetOptions():  # DetectOptions检测选项
    def __init__(self):
        self.lr = 0.004  # learningrate学习率
        self.max_epoch = 200  # 最大迭代次数
        self.batch_size = 8  # 每次参数的更新所需要的损失函数由一组数据加权得到，这组数据的数据量为batch_size
        self.num_workers = 8  # 创建多线程的数量，提前加载未来会用到的batch数据
        self.print_interval = 100  # 打印间隔，2022/01/05未搞懂该变量意思
        self.save_interval = 10  # 保存间隔，2022/01/05未搞懂该变量意思
        self.train_dir = 'D:/water/MyTest/train_imgs'  # 训练集路径
        self.train_gt_dir = 'D:/water/MyTest/train_gts'  # 训练集标签路径
        self.test_dir = 'D:/water/MyTest/test_imgs'  # 测试集路径
        self.save_dir = 'D:/water/MyTest/det_models/'  # 保存检测模型
        self.saved_model_path = 'D:/water/MyTest/det_models/checkpoint_final'  # 保存最终检测模型
        self.det_res_dir = 'D:/water/MyTest/det_res/'  # 保存测试集检测结果
        self.thresh = 0.3  # 分割后处理阈值
        self.box_thresh = 0.5  # 检测框阈值f
        self.max_candidates = 10  # 候选检测框数量（本数据集每张图像只有一个文本，因此可置为1）
        self.test_img_short_side = 640  # 测试图像最短边长度


det_args = DetOptions()


class RecOptions():
    def __init__(self):
        self.height = 32  # 图像尺寸
        self.width = 100
        self.voc_size = 21  # 字符数量 '0123456789ABCDEFGHIJ' + 'PADDING'位
        self.decoder_sdim = 512
        self.max_len = 5  # 文本长度
        self.lr = 1.0
        self.milestones = [40, 60]  # 在第 40 和 60 个 epoch 训练时降低学习率
        self.max_epoch = 200
        self.batch_size = 8
        self.num_workers = 8
        self.print_interval = 100
        self.save_interval = 10
        self.train_dir = 'D:/water/MyTest/rec_datasets/train_imgs'
        self.test_dir = 'D:/water/MyTest/rec_datasets/test_imgs'
        self.save_dir = 'D:/water/MyTest/rec_models/'
        self.saved_model_path = 'D:/water/MyTest/rec_models/checkpoint_final'
        self.rec_res_dir = 'D:/water/MyTest/rec_res/'

    def set_(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)


rec_args = RecOptions()
device = torch.device('cuda')
DEBUG = False


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.smooth = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x2, x3, x4, x5


class SegDetector(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024, 2048], inner_channels=256, k=50, bias=False):
        super(SegDetector, self).__init__()
        self.k = k
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(
            inner_channels, inner_channels // 4, 3, padding=1, bias=bias)

        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.thresh = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2),
            nn.Sigmoid())
        self.thresh.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

    def weights_init(self, m):
        # 模型权重初始化
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, features):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self.up5(in5) + in4  # 尺寸为输入图像的 1/16
        out3 = self.up4(out4) + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4

        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)

        fuse = torch.cat((p5, p4, p3, p2), 1)  # 尺寸为 batch_size，64*4， H', W'
        binary = self.binarize(fuse)
        if self.training:
            result = OrderedDict(binary=binary)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
            return result
        else:
            return binary  # for inference

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


class BasicModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.backbone = ResNet()
        self.decoder = SegDetector()

    def forward(self, data):
        output = self.backbone(data)
        output = self.decoder(output)
        return output


class DiceLoss(nn.Module):
    '''
    Loss function from https://arxiv.org/abs/1707.03237
    '''

    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask):
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape

        interp = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * interp / union
        return loss


class MaskL1Loss(nn.Module):
    def __init__(self):
        super(MaskL1Loss, self).__init__()

    def forward(self, pred, gt, mask):
        mask_sum = mask.sum()
        if mask_sum.item() == 0:
            return mask_sum
        else:
            loss = (torch.abs(pred[:, 0] - gt) * mask).sum() / mask_sum
            return loss


class BalanceCrossEntropyLoss(nn.Module):
    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self, pred, gt, mask):
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
        loss = nn.functional.binary_cross_entropy(pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
                positive_count + negative_count + self.eps)
        return balance_loss


class L1BalanceCELoss(nn.Module):
    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=1):
        super(L1BalanceCELoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()

        self.l1_scale = l1_scale  # 不同损失赋予不同权重
        self.bce_scale = bce_scale

    def forward(self, pred, batch):
        metrics = dict()
        bce_loss = self.bce_loss(pred['binary'], batch['gt'], batch['mask'])
        l1_loss = self.l1_loss(pred['thresh'], batch['thresh_map'], batch['thresh_mask'])
        dice_loss = self.dice_loss(pred['thresh_binary'], batch['gt'], batch['mask'])

        loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
        metrics['binary_loss'] = bce_loss
        metrics['thresh_loss'] = l1_loss
        metrics['thresh_binary_loss'] = dice_loss

        return loss, metrics


class SegDetectorModel(nn.Module):
    def __init__(self, device):
        super(SegDetectorModel, self).__init__()
        self.model = BasicModel()
        self.criterion = L1BalanceCELoss()
        self.device = device
        self.to(self.device)

    def forward(self, batch, training=True):
        for key, value in batch.items():
            if value is not None and hasattr(value, 'to'):
                batch[key] = value.to(self.device)

        pred = self.model(batch['image'].float())

        if self.training:
            loss, metrics = self.criterion(pred, batch)  # 计算损失函数
            return pred, loss, metrics
        else:
            return pred


class SegDetectorRepresenter():
    '''
    从 probability map 得到检测框的方法
    '''

    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=100):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.min_size = 3
        self.scale_ratio = 0.4

    def represent(self, batch, pred):
        images = batch['image']
        segmentation = pred > self.thresh  # 将预测分割图进行二值化
        boxes_batch = []
        scores_batch = []
        for batch_index in range(images.size(0)):
            height, width = batch['shape'][batch_index]
            boxes, scores = self.polygons_from_bitmap(pred[batch_index], segmentation[batch_index],
                                                      width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        assert _bitmap.size(0) == 1
        bitmap = _bitmap.cpu().numpy()[0]
        pred = pred.cpu().detach().numpy()[0]
        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)  # 找分割轮廓
        # contours, _ = cv2.findContours这是原本的代码，与2021/12/30修改
        # 于2022/03/10将_, contours, _ = cv2.findContours改为_, contours = cv2.findContours
        for contour in contours[:self.max_candidates]:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)  # 多边形拟合轮廓曲线
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = self.box_score_fast(pred, points.reshape(-1, 2))  # 计算分割区域的整体得分，去除低分候选区域
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=2.0)  # 因为得到的分割结果是文本收缩区域，因此需要进行一定程度扩张
                if len(box) != 1:
                    continue
            else:
                continue

            box = box.reshape(-1, 2)
            mini_box, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))  # 计算最小外接矩形
            if sside < self.min_size + 2:
                continue

            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            mini_box[:, 0] = np.clip(np.round(mini_box[:, 0] / width * dest_width), 0,
                                     dest_width)  # 尺寸与原图对齐
            mini_box[:, 1] = np.clip(np.round(mini_box[:, 1] / height * dest_height), 0,
                                     dest_height)
            boxes.append(mini_box.tolist())
            scores.append(score)
        return boxes, scores

    def unclip(self, box, unclip_ratio=1.5):
        '''
        做一定程度的分割区域扩张
        '''
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = np.array([points[index_1], points[index_2],
                        points[index_3], points[index_4]]).reshape(4, 2)
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        '''
        计算多边形检测区域的分数（多边形所包含的像素点预测为前景文本的分数的平均值）
        '''
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)  # 2022/3/25修改，将np.int修改为int
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)  # 2022/3/25修改，将np.int修改为int
        ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)  # 2022/3/25修改，将np.int修改为int
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)  # 2022/3/25修改，将np.int修改为int

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def resize_image(img):
    # 图像最短边设定为预设长度，长边根据原图尺寸比例进行缩放
    height, width, _ = img.shape
    if height < width:
        new_height = det_args.test_img_short_side
        new_width = int(math.ceil(new_height / height * width / 32) * 32)
    else:
        new_width = det_args.test_img_short_side
        new_height = int(math.ceil(new_width / width * height / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


def load_test_image(image_path):
    RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
    img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
    original_shape = img.shape[:2]
    img = resize_image(img)
    img -= RGB_MEAN
    img /= 255.
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    return img, original_shape


def format_output(det_res_dir, batch, output):
    batch_boxes = output
    batch_scores = output  # 2021/12/30改，原来为batch_boxes,batch_scores=output
    for index in range(batch['image'].size(0)):
        original_shape = batch['shape'][index]
        filename = batch['filename'][index]
        result_file_name = 'det_res_' + filename.split('/')[-1].split('.')[0] + '.txt'
        result_file_path = os.path.join(det_res_dir, result_file_name)
        boxes = batch_boxes[index]
        scores = batch_scores[index]
        with open(result_file_path, 'wt') as res:
            for i, box in enumerate(boxes):
                box = np.array(box).reshape(-1).tolist()
                result = ",".join([str(int(x)) for x in box])
                score = scores[i]
                res.write(result + ',' + str(score) + "\n")


def det_test():
    # 模型加载
    model = SegDetectorModel(device)
    model.load_state_dict(torch.load(det_args.saved_model_path, map_location=device), strict=False)
    model.eval()
    # 后处理
    representer = SegDetectorRepresenter(thresh=det_args.thresh, box_thresh=det_args.box_thresh,
                                         max_candidates=det_args.max_candidates)
    # 推理
    os.makedirs(det_args.det_res_dir, exist_ok=True)
    batch = dict()
    cnt = 0
    with torch.no_grad():
        for file in tqdm(os.listdir(det_args.test_dir)):
            img_path = os.path.join(det_args.test_dir, file)
            image, ori_shape = load_test_image(img_path)
            batch['image'] = image
            batch['shape'] = [ori_shape]
            batch['filename'] = [file]
            pred = model.forward(batch, training=False)
            output, temp = representer.represent(batch, pred)  # 2021/12/30改，representer.
            # represent(batch, pred)返回两个值，
            # 原代码只有一个变量接收返回值，我加了一个temp
            format_output(det_args.det_res_dir, batch, output)

            if DEBUG and cnt >= 6:  # DEBUG
                break
            cnt += 1


class _Block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(_Block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class RecBackbone(nn.Module):
    def __init__(self):
        super(RecBackbone, self).__init__()

        in_channels = 3
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.inplanes = 32
        self.layer1 = self._make_layer(32, 3, [2, 2])  # [16, 50]
        self.layer2 = self._make_layer(64, 4, [2, 2])  # [8, 25]
        self.layer3 = self._make_layer(128, 6, [2, 1])  # [4, 25]
        self.layer4 = self._make_layer(256, 6, [2, 1])  # [2, 25]
        self.layer5 = self._make_layer(512, 3, [2, 1])  # [1, 25]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != [1, 1] or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(_Block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_Block(self.inplanes, planes))
            return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        cnn_feat = x5.squeeze(2)  # [N, c, w]
        cnn_feat = cnn_feat.transpose(2, 1)

        return cnn_feat


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn=512, nHidden=512, nOut=512):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class RecModelBuilder(nn.Module):
    def __init__(self, rec_num_classes, sDim=512):
        super(RecModelBuilder, self).__init__()
        self.rec_num_classes = rec_num_classes
        self.sDim = sDim

        self.encoder = RecBackbone()
        self.decoder = nn.Sequential(
            BidirectionalLSTM(sDim, sDim, sDim),
            BidirectionalLSTM(sDim, sDim, rec_num_classes))

        self.rec_crit = nn.CTCLoss(zero_infinity=True)

    def forward(self, inputs):
        x, rec_targets, rec_lengths = inputs
        batch_size = x.shape[0]

        encoder_feats = self.encoder(x)  # N, T, C
        encoder_feats = encoder_feats.transpose(0, 1).contiguous()  # T, N, C
        rec_pred = self.decoder(encoder_feats)

        if self.training:
            rec_pred = rec_pred.log_softmax(dim=2)
            preds_size = torch.IntTensor([rec_pred.size(0)] * batch_size)
            loss_rec = self.rec_crit(rec_pred, rec_targets, preds_size, rec_lengths)
            return loss_rec
        else:
            rec_pred_scores = torch.softmax(rec_pred.transpose(0, 1), dim=2)
            return rec_pred_scores


class WMRDataset(data.Dataset):
    def __init__(self, data_dir=None, max_len=5, resize_shape=(32, 100), train=True):
        super(WMRDataset, self).__init__()
        self.data_dir = data_dir
        self.max_len = max_len
        self.is_train = train

        self.targets = [[os.path.join(data_dir, t), t.split('_')[-1][:5]] for t in
                        os.listdir(data_dir) if t.endswith('.jpg')]
        self.PADDING, self.char2id, self.id2char = self.gen_labelmap()

        # 数据增强
        self.transform = transforms.Compose([
            transforms.Resize(resize_shape),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            # 可以添加更多的数据增强操作，比如 gaussian blur、shear 等
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.targets)

    @staticmethod
    def gen_labelmap(charset='0123456789ABCDEFGHIJ'):
        # 构造字符和数字标签对应字典
        PADDING = 'PADDING'
        char2id = {t: idx for t, idx in zip(charset, range(1, 1 + len(charset)))}
        char2id.update({PADDING: 0})
        id2char = {v: k for k, v in char2id.items()}
        return PADDING, char2id, id2char

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.targets[index][0]
        word = self.targets[index][1]
        img = Image.open(img_path)

        label = np.full((self.max_len,), self.char2id[self.PADDING], dtype=np.int)
        label_list = []
        word = word[:self.max_len]
        for char in word:
            label_list.append(self.char2id[char])

        label_len = len(label_list)
        assert len(label_list) <= self.max_len
        label[:len(label_list)] = np.array(label_list)

        if self.transform is not None and self.is_train:
            img = self.transform(img)
            img.sub_(0.5).div_(0.5)

        label_len = np.array(label_len).astype(np.int32)
        label = np.array(label).astype(np.int32)

        return img, label, label_len  # 输出图像、文本标签、标签长度, 计算 CTC loss 需要后两者信息


def rec_load_test_image(image_path, size=(100, 32)):
    img = Image.open(image_path)
    img = img.resize(size, Image.BILINEAR)
    img = torchvision.transforms.ToTensor()(img)
    img.sub_(0.5).div_(0.5)
    return img.unsqueeze(0)


def rec_decode(rec_prob, labelmap, blank=0):
    raw_str = torch.max(rec_prob, dim=-1)[1].data.cpu().numpy()
    res_str = []
    for b in range(len(raw_str)):
        res_b = []
        prev = -1
        for ch in raw_str[b]:
            if ch == prev or ch == blank:
                prev = ch
                continue
            res_b.append(labelmap[ch])
            prev = ch
        res_str.append(''.join(res_b))
    return res_str


def rec_test_data_gen():
    test_dir = 'D:/water/MyTest/test_imgs'
    det_dir = 'D:/water/MyTest/det_res'
    word_save_dir = 'D:/water/MyTest/rec_datasets/test_imgs/'
    os.makedirs(word_save_dir, exist_ok=True)
    label_files = os.listdir(det_dir)
    for label_file in tqdm(label_files):
        if not label_file.endswith('.txt'):
            continue
        with open(os.path.join(det_dir, label_file), 'r') as f:
            lines = f.readlines()
        if len(lines) == 0:
            continue
        line = lines[0].strip().split(',')
        locs = [float(t) for t in line[:8]]

        # image warp
        x1, y1, x2, y2, x3, y3, x4, y4 = locs
        w = int(0.5 * (((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 + (
                (x4 - x3) ** 2 + (y4 - y3) ** 2) ** 0.5))
        h = int(0.5 * (((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5 + (
                (x4 - x1) ** 2 + (y4 - y1) ** 2) ** 0.5))
        src_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype='float32')
        dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype='float32')
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        image = cv2.imread(os.path.join(test_dir, label_file.replace('det_res_', '')[:-4] + '.jpg'))
        word_image = cv2.warpPerspective(image, M, (w, h))

        # save images
        cv2.imwrite(os.path.join(word_save_dir, label_file.replace('det_res_', '')[:-4] + '.jpg'),
                    word_image)


def rec_test():
    model = RecModelBuilder(rec_num_classes=rec_args.voc_size, sDim=rec_args.decoder_sdim)
    model.load_state_dict(torch.load(rec_args.saved_model_path, map_location=device))
    model.eval()

    os.makedirs(rec_args.rec_res_dir, exist_ok=True)
    _, _, labelmap = WMRDataset().gen_labelmap()  # labelmap是类别和字符对应的字典
    with torch.no_grad():
        for file in tqdm(os.listdir(rec_args.test_dir)):
            img_path = os.path.join(rec_args.test_dir, file)
            image = rec_load_test_image(img_path)
            batch = [image, None, None]
            pred_prob = model.forward(batch)
            # todo post precess
            rec_str = rec_decode(pred_prob, labelmap)[0]
            # write to file
            with open(os.path.join(rec_args.rec_res_dir, file.replace('.jpg', '.txt')), 'w') as f:
                f.write(rec_str)


def final_postProcess():
    SPECIAL_CHARS = {k: v for k, v in zip('ABCDEFGHIJ', '1234567890')}

    test_dir = 'D:/water/MyTest/test_imgs'
    rec_res_dir = 'D:/water/MyTest/rec_res'
    rec_res_files = os.listdir(rec_res_dir)

    final_res = dict()
    for file in os.listdir(test_dir):
        res_file = file.replace('.jpg', '.txt')
        if res_file not in rec_res_files:
            final_res[file] = ''
            continue

        with open(os.path.join(rec_res_dir, res_file), 'r') as f:
            rec_res = f.readline().strip()
        final_res[file] = ''.join(
            [t if t not in 'ABCDEFGHIJ' else SPECIAL_CHARS[t] for t in rec_res])

    with open('D:/water/MyTest/rec_res/final_res.txt', 'w') as f:
        for key, value in final_res.items():
            # f.write(key + '\t' + value + '\n')
            f.write(key + ',' + value + '\n')


####################################################################################################
db = pymysql.connect(host='localhost',
                     user='root',
                     password='12345',
                     database='water')

img_path = "D:/water/MyTest/test_imgs"
res_path = 'D:/water/MyTest/rec_res'


def pic_name():
    file_nums = sum(
        [os.path.isdir(listx) for listx in os.listdir('D:/water/MyTest/test_imgs')])
    db = pymysql.connect(host='localhost',
                         user='root',
                         password='12345',
                         database='water')
    with db:
        with db.cursor() as cursor:
            # Create a new record
            f = open('D:/water/MyTest/rec_res/final_res.txt')
            for line in f:
                linelist = line.split()
                sql = "INSERT INTO `pic_res` (`user_id`,`pic_name`, `result`) VALUES (%s,%s, %s)"
                cursor.execute(sql, (int(str(linelist[0]).split('-')[0]), linelist[0], linelist[1]))
            # connection is not autocommit by default. So you must commit to save
            # your changes.
        db.commit()


def picture():
    dir_list = os.listdir(img_path)
    db = pymysql.connect(host='localhost',
                         user='root',
                         password='12345',
                         database='water')
    with db:
        with db.cursor() as cursor:
            for i in range(0, len(dir_list)):
                fp = open(os.path.join(img_path, dir_list[i]), 'rb')
                img = fp.read()
                fp.close()
                sql = "INSERT INTO `picture` (`picture`,`pic_name`) VALUES  (%s,%s)"
                cursor.execute(sql, (img, dir_list[i]))
        db.commit()


def movefile(oripath, tardir):
    filename = os.path.basename(oripath)
    tarpath = os.path.join(tardir, filename)
    # 判断原始文件路劲是否存在
    if not os.path.exists(oripath):
        print('the dir is not exist:%s' % oripath)
        status = 0
    else:
        # 判断目标文件夹是否存在
        if os.path.exists(tardir):
            # 判断目标文件夹里原始文件是否存在，存在则删除
            if os.path.exists(tarpath):
                os.remove(tarpath)
        else:
            # 目标文件夹不存在则创建目标文件夹
            os.makedirs(tardir)
        # 移动文件
        shutil.move(oripath, tardir)
        status = 1
    return status


def move_picture():
    dir_list = os.listdir(img_path)
    for i in range(0, len(dir_list)):
        movefile(os.path.join(img_path, dir_list[i]), "D:/water/MyTest/test_imgs_bk")


def move_res():
    dir_list = os.listdir(res_path)
    for i in range(0, len(dir_list)):
        movefile(os.path.join(res_path, dir_list[i]), "D:/water/MyTest/rec_res_bk")


def remove_dir():
    shutil.rmtree('D:/water/MyTest/det_res')
    shutil.rmtree('D:/water/MyTest/rec_datasets/test_imgs')


####################################################################################################
UPLOAD_FOLDER = 'D:/water/MyTest/test_imgs'

bp = Blueprint('qa', __name__, url_prefix='/')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


@bp.route('/history')
@login_required
def history():
    db = pymysql.connect(host='localhost',
                         user='root',
                         password='12345',
                         database='water')
    with db:
        with db.cursor() as cursor:
            if g.user.auto == 1:
                sql = "SELECT pic_res.pic_name,pic_res.result FROM pic_res order by user_id"
                cursor.execute(sql)
                tu = cursor.fetchall()
                # sql = "SELECT pic_res.pic_name,pic_res.result FROM pic_res order by pic_name desc limit 2"
                # cursor.execute(sql)
                # tu = cursor.fetchall()
                # print(tu)
            else:
                sql = "SELECT pic_res.pic_name,pic_res.result FROM pic_res WHERE " \
                      "pic_res.pic_name like %s"
                cursor.execute(sql, str(g.user.id) + '%')
                tu = cursor.fetchall()
            # tu = ()
        # connection is not autocommit by default. So you must commit to save
        # your changes.
        db.commit()
    li = ['left', 'center', 'right']
    return render_template('history.html', datas=tu, lis=li)


@bp.route('/delt/<string:content>', methods=['GET', 'POST'])
def delt(content):
    if request.method == 'GET':
        # content = request.form.get('delt')
        content = content
        db = pymysql.connect(host='localhost',
                             user='root',
                             password='12345',
                             database='water')
        with db:
            with db.cursor() as cursor:
                sql = "delete from pic_res where pic_name = %s"
                cursor.execute(sql, content)
            db.commit()
        os.remove(os.path.join('E:/flaskProject/static', content))
    return redirect(url_for('qa.history'))


@bp.route('/alte/<string:content>', methods=['GET', 'POST'])
def alte(content):
    if request.method == 'GET':
        a = request.args.get('alte')
        db = pymysql.connect(host='localhost',
                             user='root',
                             password='12345',
                             database='water')

        with db:
            with db.cursor() as cursor:
                sql = "UPDATE pic_res SET result = %s WHERE pic_name = %s"
                cursor.execute(sql, (a, content))
            db.commit()
    return redirect(url_for('qa.history'))


@bp.route('/search')
@login_required
def search():
    b = request.args.get('q')
    # li = b.split(' ')
    db = pymysql.connect(host='localhost',
                         user='root',
                         password='12345',
                         database='water')
    with db:
        with db.cursor() as cursor:
            # sql = "select pic_name,result from pic_res where pic_name like %s or pic_name like %s"
            # cursor.execute(sql, ('%' + str(li[0]) + '%',
            #                      '%' + str(li[1]) + '%'))
            if g.user.auto == 0:
                sql = "select pic_name,result from pic_res where pic_name like %s and user_id = %s"
                cursor.execute(sql, ('%' + b + '%', g.user.id))
            else:
                sql = "select pic_name,result from pic_res where pic_name like %s"
                cursor.execute(sql, '%' + b + '%')
            tu = cursor.fetchall()
        db.commit()
    li = ['left', 'center', 'right']
    return render_template('history.html', datas=tu, lis=li)


@bp.route('/rec')
def re():
    det_test()  # 产生det_res文件夹
    rec_test_data_gen()  # 产生rec_datasets下的test_imgs文件夹
    rec_test()  # 产生rec_res文件夹
    final_postProcess()  # 产生final_res.txt文件
    pic_name()
    picture()
    move_picture()
    move_res()
    remove_dir()
    flash('识别完成')
    return redirect('/tmp')


@bp.route('/', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('请选择图片')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('没有选择图片')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash(f'请选择{config.ALLOWED_EXTENSIONS}格式的图片')
            return redirect(url_for('qa.upload_file'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filename1 = str(str(g.user.id) + '-' + str(datetime.now()))
            filename2 = filename1.replace(' ', '-')
            filename3 = filename2.replace(':', '-')
            filename4 = filename3.replace('.', '-')
            filename5 = filename4 + '.' + filename.rsplit('.', 1)[1].lower()
            os.rename(os.path.join(UPLOAD_FOLDER, filename),
                      os.path.join(UPLOAD_FOLDER, filename5))
            shutil.copyfile(os.path.join(UPLOAD_FOLDER, filename5),
                            os.path.join('E:/flaskProject/static', filename5))
            flash('上传成功')
            return redirect('/')
    return render_template('index.html')


@bp.route('/tmp', methods=['GET', 'POST'])
@login_required
def upload_file1():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('请选择图片')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('没有选择图片')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash(f'请选择{config.ALLOWED_EXTENSIONS}格式的图片')
            return redirect(url_for('qa.upload_file1'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            # filename1 = str(str(g.user.id) + '-' + str(datetime.now()))
            # filename2 = filename1.replace(' ', '-')
            # filename3 = filename2.replace(':', '-')
            # filename4 = filename3.replace('.', '-')
            # filename5 = filename4 + '.' + filename.rsplit('.', 1)[1].lower()
            # os.rename(os.path.join(UPLOAD_FOLDER, filename),
            #           os.path.join(UPLOAD_FOLDER, filename5))
            # shutil.copyfile(os.path.join(UPLOAD_FOLDER, filename5),
            #                 os.path.join('E:/flaskProject/static', filename5))
            shutil.copyfile(os.path.join(UPLOAD_FOLDER, filename),
                            os.path.join('E:/flaskProject/static', filename))
            flash('上传成功')
            return redirect('/tmp')
    return render_template('index1.html')
