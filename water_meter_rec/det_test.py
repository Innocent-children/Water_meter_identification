import os
import torch
import numpy as np
import cv2
import math
import pyclipper
from shapely.geometry import Polygon
from tqdm import tqdm

# 是否使用 GPU
import det_train

device = torch.device('cuda')
DEBUG = False


# 参数设置
class DetOptions():  # DetectOptions检测选项
    def __init__(self):
        self.lr = 0.004  # learningrate学习率
        self.max_epoch = 200  # 最大迭代次数
        self.batch_size = 8  # 每次参数的更新所需要的损失函数由一组数据加权得到，这组数据的数据量为batch_size
        self.num_workers = 8  # 创建多线程的数量，提前加载未来会用到的batch数据
        self.print_interval = 100  # 打印间隔，2022/01/05未搞懂该变量意思
        self.save_interval = 10  # 保存间隔，2022/01/05未搞懂该变量意思
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


# 从分割图得到最终文字坐标的后处理方法
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
        # contours, _ = cv2.findContours这是原本的代码，于2021/12/30修改
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
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


# 测试图片处理
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


# 检测结果输出
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

            # 测试


def det_test():
    # 模型加载
    model = det_train.SegDetectorModel(device)
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
