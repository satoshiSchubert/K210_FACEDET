# K210_FACEDET
2020年全国大学生电子设计竞赛F题视觉部分解决方案

FACEDET-K210

<a href="https://996.icu"><img src="https://img.shields.io/badge/link-996.icu-red.svg" alt="996.icu" /></a>

last Update:2020/12/20
|Author|crisprhhx|
|---|---
|E-mail|crisprhhx@outlook.com
|---|crisprhhx@qq.com

## Introduction
2020TI杯全国大学生电子设计大赛F题解决方案视觉部分。
代码可以实现三个功能，分别是人脸识别，现场学习，和口罩识别。通过按下外接的按键来给芯片一个触发信号以切换识别模式。

人脸识别主要是通过用神经网络作为编码器，将人脸图像的特征提取出来并储存；检测时将陌生人的人脸图像计算后得到的特征与数据库中的特征进行比对；若误差小于一定阈值，则判定是同一人。
口罩识别也是运用了深度学习，通过训练几千张的（图中的人是否有戴口罩）的数据集，生成一个检测是否佩戴口罩的模型。一般来说模型训练多在笔记本或台式机上完成，k210通过调用训练出来的模型文件来识别。

相关原理可以百度或谷歌，这些技术目前都已经非常成熟，网上有海量的资料可供学习，这里不再赘述。

## Setup
0. 首先你需要一块K210开发板

1. 根据TUITION.jpg中的内容操作，配置环境；文件夹中两个bin文件是固件，因为后续步骤加载模型后会很占内存，因此需要切换成更小的固件（maixpy_dls_mini_with_openmv.bin）以腾出空间。

2. 在IDE中加载Release文件夹中的文件，其中main.py为源代码；其余三个为权重文件。

3. 若需要实现脱离笔记本在k210上运行算法，需要准备一张SD卡，并将三个权重文件放进去。运行时k210将会从SD卡中加载模型到内存中。

Note：如果出现无法检测到sensor的情况，可能是因为摄像头接触不良。需要调整一下摄像头的位置或者重新插拔摄像头。

## Usage
先连着电脑运行，有效果之后可以尝试将代码下载到芯片之后独立运行。
代码中已经分配了引脚。可以参照k210官方文档学习一下GPIO的设置方式（比单片机简单得多，k210内置FPGA，可以任意映射引脚），并以此配置外接电路。

## Train
如何训练出K210可以使用的权重文件：
https://www.bilibili.com/video/av925105825?share_medium=android&share_source=qq&bbid=A2ECE591-E959-4654-B2E6-D29EBAEACD6615220infoc&ts=1602259328125

## Support
官方交流群（有问题都可以在群里问，很快就会有解答）：
荔枝 MaixPy AI 一群 878189804
email：crisprhhx@outlook.com
	  crisprhhx@qq.com


