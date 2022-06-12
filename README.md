# Water_meter_identification——一个关于字轮式水表识别的项目
## water_meter_rec

这是水表识别的主体代码

输入一张水表图片，会输出该水表的数据

主要分为文字检测以及文本识别

文字检测部分采用的是DBnet，其中CNN部分采取的是ResNet-50网络，经过测试，效果不错

文本识别部分采用的是CRNN，其中特征提取部分采取的是类ResNet网络，经过测试，效果不错

## flaskProject

这是web系统的代码

web系统配合water_meter_rec使用，在flaskProject中引入water_meter_rec产生的训练结果文件，可以在web端进行操作。
