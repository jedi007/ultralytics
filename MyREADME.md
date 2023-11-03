https://github.com/ultralytics/ultralytics 官方教程：https://docs.ultralytics.com/modes/train/

资源安装

更建议下载代码后使用 下面指令安装，这样可以更改源码，如果不需要更改源码就直接pip install ultralytics也是可以的。

需要到ultralytics目录下执行以下指令
pip install -e .

这样安装后，可以直接修改yolov8源码，并且可以立即生效。此图是命令解释： 安装成功后： pip list可以看到安装的包：





detect
export pred shape: shape(1,84,6300)  :  (batch_size, [xywh + numclasses], numboxes)
train pred shape: 
[torch.Size([1, 144, 80, 80]), torch.Size([1, 144, 40, 40]), torch.Size([1, 144, 20, 20])]  三个尺boxes下的输出
转化为： torch.Size([1, 64, 8400]) , torch.Size([1, 80, 8400]) 
前者是（batch_szize, 16*4, 80*80+40*40+20*20） 8400是3尺度合一的boxes数量,  16*4 是4个坐标数据：(x1,y1)(x2,y2), 16是格子数，按照分类的思想，x1预测从0-15的偏移，然后加权就和做为其预测值
后者是 (batch_size, num_classes, 80*80+40*40+20*20)