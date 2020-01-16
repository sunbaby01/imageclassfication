图像分类器

使用方法：

训练阶段：

1.首先运行 imageclassfication_train.py

2.输入数据集地址（数据集地址内包含n个文件夹分别是n个标签的图片）你可以使用crawler.py进行收集图片

3.输入模型名 （如resnet50）

4.等待运行完成 （也可中途中断 每轮自动保存当前模型）


使用阶段：

1.首先运行 imageclassfication_test.py

2.输入图片文件夹地址 

3.得到结果
