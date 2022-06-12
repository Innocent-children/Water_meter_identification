import os
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt

# 是否使用 GPU
device = torch.device('cuda')
DEBUG = False


class RecOptions():
    def __init__(self):
        self.height = 32  # 图像尺寸
        self.width = 100
        self.voc_size = 21  # 字符数量 '0123456789ABCDEFGHIJ' + 'PADDING'位
        self.decoder_sdim = 512
        self.max_len = 5  # 文本长度
        self.lr = 0.5  # 原为1.0，未改动
        self.milestones = [40, 60]  # 在第 40 和 60 个 epoch 训练时降低学习率
        self.max_epoch = 80  # 原为80，改为200
        self.batch_size = 32  # 原为64，改为8
        self.num_workers = 8
        self.print_interval = 25  # 原为25，改为100
        self.save_interval = 125  # 原为125，改为10
        self.train_dir = '/MyTest/rec_datasets/train_imgs'
        self.test_dir = '/MyTest/rec_datasets/test_imgs'
        self.save_dir = '/MyTest/rec_models/'
        self.saved_model_path = '/MyTest/rec_models/checkpoint_final'
        self.rec_res_dir = '/MyTest/rec_res/'

    def set_(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)


rec_args = RecOptions()

'''
数据集导入方法
'''


# data
class WMRDataset(data.Dataset):
    def __init__(self, data_dir=None, max_len=5, resize_shape=(32, 100), train=True):
        super(WMRDataset, self).__init__()
        self.data_dir = data_dir
        self.max_len = max_len
        self.is_train = train

        self.targets = [[os.path.join(data_dir, t), t.split('_')[-1][:5]] for t in
                        os.listdir(data_dir) if t.endswith('.jpg')]  # 从train_imgs的图片名字中提取出图片结果当做标签
        self.PADDING, self.char2id, self.id2char = self.gen_labelmap()

        # 数据增强
        self.transform = transforms.Compose([
            transforms.Resize(resize_shape),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            # 可以添加更多的数据增强操作，比如 gaussian blur、shear 等
            # transforms.GaussianBlur(kernel_size=3),  # 自己添加
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


dataset = WMRDataset(rec_args.train_dir, max_len=5, resize_shape=(rec_args.height, rec_args.width),
                     train=True)
train_dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True,
                                   drop_last=False)
batch = next(iter(train_dataloader))

image, label, label_len = batch
image = ((image[0].permute(1, 2, 0).to('cpu').numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
plt.title('image')
plt.xticks([])
plt.yticks([])
plt.imshow(image)

label_digit = label[0].to('cpu').numpy().tolist()
label_str = ''.join([dataset.id2char[t] for t in label_digit if t > 0])

print('label_digit: ', label_digit)
print('label_str: ', label_str)


# backbone
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


# decoder
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


# basic
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


'''
模型各阶段数据结构展示
'''
dataset = WMRDataset(rec_args.train_dir, max_len=rec_args.max_len,
                     resize_shape=(rec_args.height, rec_args.width), train=True)
train_dataloader = data.DataLoader(dataset, batch_size=2, num_workers=0, shuffle=True,
                                   pin_memory=True, drop_last=False)
batch = next(iter(train_dataloader))

model = RecModelBuilder(rec_num_classes=rec_args.voc_size, sDim=rec_args.decoder_sdim)
model = model.to(device)
model.train()

image, rec_targets, rec_lengths = [v.to(device) for v in batch]
encoder_out = model.encoder(image)
decoder_out = model.decoder(encoder_out.transpose(0, 1).contiguous())

# batch 输入
print('batch 输入 [image, label, label_length]：')
print(batch[0].shape)
print(batch[1].shape)
print(batch[2].shape)
print()

# encoder 输出
print('encoder 输出：')
print(encoder_out.shape)
print()

# decoder 输出
print('decoder 输出：')
print(decoder_out.shape)


# train
def rec_train():
    # dataset
    dataset = WMRDataset(rec_args.train_dir, max_len=rec_args.max_len,
                         resize_shape=(rec_args.height, rec_args.width), train=True)
    train_dataloader = data.DataLoader(dataset, batch_size=rec_args.batch_size,
                                       num_workers=rec_args.num_workers, shuffle=True,
                                       pin_memory=True, drop_last=False)

    # model
    model = RecModelBuilder(rec_num_classes=rec_args.voc_size, sDim=rec_args.decoder_sdim)
    model = model.to(device)
    model.train()

    # Optimizer    优化器
    param_groups = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adadelta(param_groups, lr=rec_args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=rec_args.milestones,
                                                     gamma=0.1)

    os.makedirs(rec_args.save_dir, exist_ok=True)
    # do train
    step = 0
    # if __name__ == '__main__':  # 2021/12/30自己添加该行代码，解决报错
    for epoch in range(rec_args.max_epoch):
        current_lr = optimizer.param_groups[0]['lr']

        for i, batch in enumerate(train_dataloader):
            step += 1
            batch = [v.to(device) for v in batch]
            loss = model(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print
            if step % rec_args.print_interval == 0:
                print(
                    'step: {:4d}\tepoch: {:4d}\tloss: {:.4f}'.format(step, epoch, loss.item()))
        scheduler.step()

        # save
        if epoch % rec_args.save_interval == 0:
            save_name = 'checkpoint_' + str(epoch)
            torch.save(model.state_dict(), os.path.join(rec_args.save_dir, save_name))

    torch.save(model.state_dict(), rec_args.saved_model_path)
