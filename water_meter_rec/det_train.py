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
from shapely.geometry import Polygon
from collections import OrderedDict

# 是否使用 GPU
device = torch.device('cuda')
DEBUG = False


# 参数设置
class DetOptions():  # DetectOptions检测选项
    def __init__(self):
        self.lr = 0.004  # learningrate学习率
        self.max_epoch = 200  # 最大迭代次数
        self.batch_size = 8  # 每次参数的更新所需要的损失函数由一组数据加权得到，这组数据的数据量为batch_size
        self.num_workers = 8  # 创建多线程的数量，提前加载未来会用到的batch数据
        self.print_interval = 100  # 打印间隔
        self.save_interval = 10  # 保存间隔
        self.train_dir = '/MyTest/train_imgs'  # 训练集路径
        self.train_gt_dir = '/MyTest/train_gts'  # 训练集标签路径
        self.test_dir = '/MyTest/test_imgs'  # 测试集路径
        self.save_dir = '/MyTest/det_models/'  # 保存检测模型
        self.saved_model_path = '/MyTest/det_models/checkpoint_final'  # 保存最终检测模型
        self.det_res_dir = '/MyTest/det_res/'  # 保存测试集检测结果
        self.thresh = 0.3  # 分割后处理阈值
        self.box_thresh = 0.5  # 检测框阈值f
        self.max_candidates = 10  # 候选检测框数量（本数据集每张图像只有一个文本，因此可置为1）
        self.test_img_short_side = 640  # 测试图像最短边长度


det_args = DetOptions()

'''
数据增强变换方法
'''


class BaseAugment():
    '''
    通过 imgaug.augmenters 进行基础变换，包括尺寸调整、翻转、旋转等
    '''

    def __init__(self, only_resize=False, keep_ratio=False, augmenters=None, resize_shape=None):
        self.only_resize = only_resize
        self.keep_ratio = keep_ratio
        self.augmenter = augmenters
        self.resize_shape = resize_shape

    def resize_image(self, image):  # 改变图像大小
        origin_height, origin_width, _ = image.shape  # 将图像最初的大小保存下来
        height = self.resize_shape['height']
        width = self.resize_shape['width']
        if self.keep_ratio:  # 是否保持图像长宽比不变
            width = origin_width * height / origin_height
            N = math.ceil(width / 32)
            width = N * 32
        image = cv2.resize(image, (width, height))
        return image

    def process(self, data):
        image = data['image']
        shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            if self.only_resize:
                data['image'] = self.resize_image(image)  # 只进行尺寸调整
            else:
                data['image'] = aug.augment_image(image)  # 图像变换
            self.may_augment_annotation(aug, data, shape)  # 对 polygon 标注进行对应的变换

        filename = data.get('filename', data.get('data_id', ''))
        data.update(filename=filename, shape=shape[:2])
        return data

    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for line in data['lines']:
            if self.only_resize:
                new_poly = [(p[0], p[1]) for p in line['poly']]
            else:
                new_poly = self.may_augment_poly(aug, shape, line['poly'])
            line_polys.append({
                'points': new_poly,
                'ignore': line['text'] == '###',  # 图像是否是困难样本（模糊不可辨），本任务数据集中不存在困难样本
                'text': line['text'],
            })
        data['polys'] = line_polys
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints([imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[
            0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly


class ColorJitter():
    '''
    颜色增强，包括亮度、对比度、饱和度、色相变换
    '''

    def __init__(self, b=0.2, c=0.2, s=0.15, h=0.15):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=b, contrast=c, saturation=s, hue=h)

    def process(self, data):
        img = data['image']
        image = Image.fromarray(img.astype('uint8')).convert('RGB')  # 数据类型转换
        img = np.array(self.color_jitter(image)).astype(np.float64)
        data['image'] = img
        return data


class RandomCropData():
    '''
    随机裁剪图像，并保证裁剪时不切割到图像中的文字区域
    '''

    def __init__(self, size=(640, 640)):
        self.size = size
        self.max_tries = 10  # 裁剪尝试的最大次数（因为存在裁剪区域太小等裁剪失败情况）
        self.min_crop_side_ratio = 0.1  # 裁剪区域边长最小比例，即裁剪的图像边长与原始图像边长的比值不能
        # 小于 min_crop_side_ratio

    def process(self, data):
        img = data['image']

        ori_img = img
        ori_lines = data['polys']
        all_care_polys = [line['points'] for line in data['polys'] if not line['ignore']]
        crop_x, crop_y, crop_w, crop_h = self.crop_area(img,
                                                        all_care_polys)  # 裁剪区域的左上角坐标(x, y)
        # 以及区域宽高(w, h)

        # 根据裁剪区域参数对图像进行裁剪，并填充空白以得到指定 size 的图像（在右侧或者底侧进行填充）
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        padimg = np.zeros((self.size[1], self.size[0], img.shape[2]), img.dtype)
        padimg[:h, :w] = cv2.resize(img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
        img = padimg

        # 根据裁剪区域参数对文字位置坐标进行转换
        lines = []
        for line in data['polys']:
            poly = ((np.array(line['points']) - (crop_x, crop_y)) * scale).tolist()
            if not self.is_poly_outside_rect(poly, 0, 0, w, h): lines.append(
                {**line, 'points': poly})  # 不保留裁剪区域之外的文字

        data['polys'] = lines
        data['image'] = img

        return data

    def is_poly_outside_rect(self, poly, x, y, w, h):
        # 判断文字polygon 是否在矩形区域外
        poly = np.array(poly)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def split_regions(self, axis):
        # 返回可划切割线的连续区域
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i - 1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    def random_select(self, axis, max_size):
        # 从一块连续区域中选择两条切割线
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    def region_wise_random_select(self, regions, max_size):
        # 从两块连续区域中选择两条切割线
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)
        return xmin, xmax

    def crop_area(self, img, polys):
        # 裁剪区域
        h, w, _ = img.shape
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        for points in polys:
            points = np.round(points, decimals=0).astype(np.int32)
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            h_array[miny:maxy] = 1
        # h_array == 1 的位置表示有文本，h_array == 0 的位置表示无文本；w_array 同理
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        for i in range(self.max_tries):
            if len(w_regions) > 1:
                # 有多块可切割区域时
                xmin, xmax = self.region_wise_random_select(w_regions, w)
            else:
                # 只有一块可切割区域时
                xmin, xmax = self.random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self.random_select(h_axis, h)

            if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:
                # 切割区域太小，不可取
                continue
            num_poly_in_rect = 0

            # 保证至少有一个文字区域在切割出的区域中即可
            for poly in polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        return 0, 0, w, h


class MakeSegDetectionData():
    '''
    构造文本区域二值图（DB论文中的 probability map），以及用于计算loss的mask
    '''

    def __init__(self, min_text_size=8, shrink_ratio=0.4):
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio  # polygon 收缩比例

    def process(self, data):
        # 数据结构调整统一，方便后续操作
        polygons = []
        ignore_tags = []
        annotations = data['polys']
        for annotation in annotations:
            polygons.append(np.array(annotation['points']))
            ignore_tags.append(annotation['ignore'])
        ignore_tags = np.array(ignore_tags, dtype=np.uint8)
        filename = data.get('filename', data['data_id'])
        shape = np.array(data['shape'])
        data = OrderedDict(image=data['image'],
                           polygons=polygons,
                           ignore_tags=ignore_tags,
                           shape=shape,
                           filename=filename,
                           is_training=data['is_training'])

        image = data['image']
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']
        image = data['image']
        filename = data['filename']

        h, w = image.shape[:2]
        if data['is_training']:
            polygons, ignore_tags = self.validate_polygons(polygons, ignore_tags, h, w)
        gt = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(polygons)):
            polygon = polygons[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            if ignore_tags[i] or min(height, width) < self.min_text_size:  # 文本区域太小时，作为困难样本 ignore
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                # 收缩 polygon 并绘制 probability map
                polygon_shape = Polygon(polygon)
                distance = polygon_shape.area * (
                        1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
                subject = [tuple(l) for l in polygons[i]]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)
                if shrinked == []:
                    cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)

        if filename is None:
            filename = ''
        data.update(image=image,
                    polygons=polygons,
                    gt=gt, mask=mask, filename=filename)
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        '''
        统一polygon坐标顺序，并且ignore面积为0的polygons
        '''
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]  # 调整坐标顺序
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * (
                    polygon[next_index, 1] + polygon[i, 1])

        return edge / 2.


class MakeBorderMap():
    '''
    构造文本边界二值图（DB论文中的 threshold map），以及用于计算loss的mask
    '''

    def __init__(self, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7):
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        warnings.simplefilter("ignore")

    def process(self, data):
        image = data['image']
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']
        canvas = np.zeros(image.shape[:2], dtype=np.float32)
        mask = np.zeros(image.shape[:2], dtype=np.float32)

        for i in range(len(polygons)):
            if ignore_tags[i]:
                continue
            self.draw_border_map(polygons[i], canvas, mask=mask)  # 绘制 border map
        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min
        data['thresh_map'] = canvas
        data['thresh_mask'] = mask
        return data

    def draw_border_map(self, polygon, canvas, mask):
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width),
                             (height, width))
        ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1),
                             (height, width))

        distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid - ymin:ymax_valid - ymax + height,
                xmin_valid - xmin:xmax_valid - xmax + width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

    def distance(self, xs, ys, point_1, point_2):
        # 计算图像中的点到 文字polygon 边界的距离，画阈值图会用到
        height, width = xs.shape[:2]
        square_distance_1 = np.square(
            xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(
            xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(
            point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / (
                2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)

        result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
        return result


class NormalizeImage():
    '''
    将图像元素值归一化到[-1, 1]
    '''
    RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])

    def process(self, data):
        assert 'image' in data, '`image` in data is required by this process'
        image = data['image']
        image -= self.RGB_MEAN
        image /= 255.
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        data['image'] = image
        return data

    @classmethod
    def restore(self, image):
        image = image.permute(1, 2, 0).to('cpu').numpy()
        image = image * 255.
        image += self.RGB_MEAN
        image = image.astype(np.uint8)
        return image


class FilterKeys():
    '''
    过滤掉后续不需要的键值对
    '''

    def __init__(self, superfluous):
        self.superfluous_keys = set(superfluous)

    def process(self, data):
        for key in self.superfluous_keys:
            del data[key]
        return data


# 训练集数据处理
train_processes = [
    BaseAugment(only_resize=False, keep_ratio=False,
                augmenters=iaa.Sequential([
                    iaa.Fliplr(0.5),  # 水平翻转
                    iaa.Affine(rotate=(-10, 10)),  # 旋转
                    iaa.Resize((0.5, 3.0))  # 尺寸调整
                ])),
    ColorJitter(),  # 颜色增强
    RandomCropData(size=[640, 640]),  # 随机裁剪
    MakeSegDetectionData(),  # 构造 probability map
    MakeBorderMap(),  # 构造 threshold map
    NormalizeImage(),  # 归一化
    FilterKeys(superfluous=['polygons', 'filename', 'shape', 'ignore_tags', 'is_training']),
    # 过滤多余元素
]

'''
数据集导入方法
'''


# dataset
class ImageDataset(data.Dataset):
    def __init__(self, data_dir=None, gt_dir=None, is_training=True, processes=None):
        self.data_dir = data_dir
        self.gt_dir = gt_dir
        self.is_training = is_training
        self.processes = processes

        self.image_paths = []
        self.gt_paths = []

        image_list = os.listdir(self.data_dir)
        self.image_paths = [self.data_dir + '/' + t for t in image_list]
        self.gt_paths = [self.gt_dir + '/' + t.replace('.jpg', '.txt') for t in image_list]
        self.targets = self.load_ann()  # 导入标注信息 一个字典，poly是坐标，text是水表内容

    def load_ann(self):
        res = []
        for gt in self.gt_paths:
            lines = []
            reader = open(gt, 'r').readlines()
            for line in reader:
                item = {}
                line = line.strip().split()
                poly = np.array(list(map(float, line[:8]))).reshape((-1, 2)).tolist()
                item['poly'] = poly
                item['text'] = line[8]  # 前8为 polygon 坐标，第9是文本字符串
                lines.append(item)
            res.append(lines)
        return res

    def __getitem__(self, index):
        if index >= len(self.image_paths):
            index = index % len(self.image_paths)
        data = {}
        image_path = self.image_paths[index]

        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        target = self.targets[index]

        data['filename'] = image_path.split('/')[-1]
        data['data_id'] = image_path.split('/')[-1]
        data['image'] = img
        data['lines'] = target
        data['is_training'] = self.is_training
        if self.processes is not None:
            for data_process in self.processes:  # 做数据增强
                data = data_process.process(data)
        return data

    def __len__(self):
        return len(self.image_paths)


# 数据处理可视化
train_dataset = ImageDataset(data_dir=det_args.train_dir, gt_dir=det_args.train_gt_dir,
                             is_training=True, processes=train_processes)
train_dataloader = data.DataLoader(train_dataset, batch_size=2, num_workers=0, shuffle=True,
                                   drop_last=False)
batch = next(iter(train_dataloader))  # 获取一个 batch


# 画图              2022/03/12修改  没啥用，注释掉
# plt.figure(figsize=(60, 60))
# image = NormalizeImage.restore(batch['image'][0])
# plt.subplot(141)
# plt.title('image', fontdict={'size': 60})
# plt.xticks([])
# plt.yticks([])
# plt.imshow(image)
#
# probability_map = (batch['gt'][0].to('cpu').numpy() * 255).astype(np.uint8)
# plt.subplot(142)
# plt.title('probability_map', fontdict={'size': 60})
# plt.xticks([])
# plt.yticks([])
# plt.imshow(probability_map, cmap='gray')
#
# threshold_map = (batch['thresh_map'][0].to('cpu').numpy() * 255).astype(np.uint8)
# plt.subplot(143)
# plt.title('threshold_map', fontdict={'size': 60})
# plt.xticks([])
# plt.yticks([])
# plt.imshow(threshold_map, cmap='gray')
#
# threshold_mask = (batch['thresh_mask'][0].to('cpu').numpy() * 255).astype(np.uint8)
# plt.subplot(144)
# plt.title('threshold_mask', fontdict={'size': 60})
# plt.xticks([])
# plt.yticks([])
# plt.imshow(threshold_mask, cmap='gray')


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# BCE loss 计算 probability map 损失
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


# MaskL1Loss 计算 threshold map 损失
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


# DiceLoss 计算 approximate binary map 损失
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


# 计算总的loss
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


'''
Backbone:采用 ResNet-50 提取金字塔式特征
'''


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


'''
Decoder:对金字塔式特征进行上采样并集联，预测输出 probability map、threshold map 以及 approximate binary map
'''


class SegDetector(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024, 2048], inner_channels=256, k=50, bias=False):
        super(SegDetector, self).__init__()
        self.k = k
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)  # in_channels:输入图像通道数
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)  # 卷积核尺寸为1x1
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)  # bias为偏差
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


# 包装
class BasicModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.backbone = ResNet()  # 对应DBnet论文从1/2到1/32那一部分
        self.decoder = SegDetector()  # 对应DBnet论文上采样级联,以及预测三个图

    def forward(self, data):
        output = self.backbone(data)
        output = self.decoder(output)
        return output


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


train_dataset = ImageDataset(data_dir=det_args.train_dir, gt_dir=det_args.train_gt_dir,
                             is_training=True, processes=train_processes)
train_dataloader = data.DataLoader(train_dataset, batch_size=2, num_workers=0, shuffle=True,
                                   drop_last=False)
batch = next(iter(train_dataloader))

# 模型各阶段数据格式
for key, value in batch.items():
    if value is not None and hasattr(value, 'to'):
        batch[key] = value.to(device)

model = SegDetectorModel(device)
model.train()
out1 = model.model.backbone(batch['image'].float())
out2 = model.model.decoder(out1)
loss, metrics = model.criterion(out2, batch)

# batch 输入
print('batch 输入：')
for key, value in batch.items():
    print(key, value.shape)
print()

# backbone 输出
print('backbone 输出：')
for o1 in out1:
    print(o1.shape)
print()

# decoder 输出
print('decoder 输出：')
for key, value in out2.items():
    print(key, value.shape)
print()

# criterion 输出
print('criterion 输出：')
print('loss: ', loss)
print(metrics)
print()


# 画图         2022/03/12修改  没啥用，注释掉
# plt.figure(figsize=(60, 60))
# image = NormalizeImage.restore(batch['image'][0])
# plt.subplot(141)
# plt.title('image', fontdict={'size': 60})
# plt.xticks([])
# plt.yticks([])
# plt.imshow(image)
#
# probability_map = (batch['gt'][0].to('cpu').numpy() * 255).astype(np.uint8)
# plt.subplot(142)
# plt.title('probability_map', fontdict={'size': 60})
# plt.xticks([])
# plt.yticks([])
# plt.imshow(probability_map, cmap='gray')
#
# threshold_map = (batch['thresh_map'][0].to('cpu').numpy() * 255).astype(np.uint8)
# plt.subplot(143)
# plt.title('threshold_map', fontdict={'size': 60})
# plt.xticks([])
# plt.yticks([])
# plt.imshow(threshold_map, cmap='gray')
#
# # 模型预测的 approximate map
# binary = (out2['binary'][0][0].detach().to('cpu').numpy() * 255).astype(np.uint8)
# plt.subplot(144)
# plt.title('binary', fontdict={'size': 60})
# plt.xticks([])
# plt.yticks([])
# plt.imshow(binary, cmap='gray')


# 学习率调整方法
class DecayLearningRate():
    def __init__(self, lr=0.004, epochs=200, factor=0.9):
        self.lr = lr
        self.epochs = epochs
        self.factor = factor

    def get_learning_rate(self, epoch):
        # 学习率随着训练过程进行不断下降
        rate = np.power(1.0 - epoch / float(self.epochs + 1), self.factor)
        return rate * self.lr

    def update_learning_rate(self, optimizer, epoch):
        lr = self.get_learning_rate(epoch)
        for group in optimizer.param_groups:
            group['lr'] = lr
        return lr


# 检测模型训练
def det_train():
    # model
    model = SegDetectorModel(device)

    # data_loader
    train_dataset = ImageDataset(data_dir=det_args.train_dir, gt_dir=det_args.train_gt_dir,
                                 is_training=True, processes=train_processes)
    train_dataloader = data.DataLoader(train_dataset, batch_size=det_args.batch_size,
                                       num_workers=det_args.num_workers, shuffle=True,
                                       drop_last=False)

    # initialize dataloader and scheduler 初始化数据加载器和调度器
    optimizer = torch.optim.SGD(model.parameters(), lr=det_args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = DecayLearningRate(lr=det_args.lr, epochs=det_args.max_epoch)

    step = 0
    epoch = 0
    model.train()
    os.makedirs(det_args.save_dir, exist_ok=True)
    for epoch in range(det_args.max_epoch):
        for batch in train_dataloader:
            step += 1
            current_lr = scheduler.update_learning_rate(optimizer, epoch)  # 学习率调整

            optimizer.zero_grad()
            pred, loss, metrics = model.forward(batch, training=True)
            loss.backward()
            optimizer.step()

            if step % det_args.print_interval == 0:
                line = []
                for key, l_val in metrics.items():
                    line.append('{0}: {1:.4f}'.format(key, l_val.mean()))
                line = '\t'.join(line)
                info = '\t'.join(
                    ['step:{:6d}', 'epoch:{:3d}', 'loss: {:4f}', '{}', 'lr: {:.4f}']).format(step,
                                                                                             epoch,
                                                                                             loss.item(),
                                                                                             line,
                                                                                             current_lr)
                print(info)
                if DEBUG:  # DEBUG 这里只跑少量数据
                    break

        # 保存阶段模型
        if epoch % det_args.save_interval == 0:
            save_name = 'checkpoint_' + str(epoch)
            torch.save(model.state_dict(), os.path.join(det_args.save_dir, save_name))
        if DEBUG:
            break

    # 保存最终模型
    torch.save(model.state_dict(), det_args.saved_model_path)
