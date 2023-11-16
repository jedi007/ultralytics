https://github.com/ultralytics/ultralytics 官方教程：https://docs.ultralytics.com/modes/train/

资源安装

更建议下载代码后使用 下面指令安装，这样可以更改源码，如果不需要更改源码就直接pip install ultralytics也是可以的。

需要到ultralytics目录下执行以下指令
pip install -e .

这样安装后，可以直接修改yolov8源码，并且可以立即生效。此图是命令解释： 安装成功后： pip list可以看到安装的包：





detect:
export pred shape: shape(1,84,6300)  :  (batch_size, [xywh + numclasses], numboxes)

train pred shape: 
[torch.Size([1, 144, 80, 80]), torch.Size([1, 144, 40, 40]), torch.Size([1, 144, 20, 20])]  三个尺寸boxes下的输出
转化为： torch.Size([1, 64, 8400]) , torch.Size([1, 80, 8400]) 
前者是（batch_szize, 16*4, 80*80+40*40+20*20） 8400是3尺度合一的boxes数量,  16*4 是4个坐标数据：(x1,y1)(x2,y2), 16是格子数，按照分类的思想，x1预测从0-15的偏移，然后加权就和做为其预测值
后者是 (batch_size, num_classes, 80*80+40*40+20*20)

train label:
17 0.324291 0.64808 0.219711 0.3164
0 0.620039 0.5939 0.172415 0.14608
五个值分别对应： label x_center y_center width height




pose:
pose train preds : ([torch.Size([1, 65, 80, 80]), torch.Size([1, 65, 40, 40]), torch.Size([1, 65, 20, 20])],  torch.Size([1, 51, 8400]))
        #  torch.Size([1, 65, 80, 80])  : (batchsize, reg_max * 4 + num_classes, grid, grid)
        #  torch.Size([1, 51, 8400]):  (batchsize, keypoints(17*3), 80*80+40*40+20*20)
train label:
0 0.671279 0.617945 0.645759 0.726859 0.519751 0.381250 2.000000 0.550936 0.348438 2.000000 0.488565 0.367188 2.000000 0.642412 0.354687 2.000000 0.488565 0.395313 2.000000 0.738046 0.526563 2.000000 0.446985 0.534375 2.000000 0.846154 0.771875 2.000000 0.442827 0.812500 2.000000 0.925156 0.964063 2.000000 0.507277 0.698438 2.000000 0.702703 0.942187 2.000000 0.555094 0.950000 2.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
前5个数的意义同datect: 分别对应： label x_center y_center width height
后面的51个的意义一次是x1,y1,type,x2,y2,type ...... x17,y17,type  ,label是person时关键点数量定义是17个。type 的值是0，1，2， 0表示没有显露出，不可见 1表示被遮挡不可见 2表示可见



pose  pred preds: preds: 
[torch.Size([1, 56, 6300]),  ([torch.Size([1, 65, 80, 60]), torch.Size([1, 65, 40, 30]), torch.Size([1, 65, 20, 15])], torch.Size([1, 51, 6300]))]

只取了  preds[0]: torch.Size([1, 56, 6300])  : (batchsize,  xywh + conf + keypoints(17*3))

经nms 变成 # preds: [torch.Size([4, 57])]  (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).  数组数量应该是batchsize

nms 中通过一下代码改变了shape：  # 从（xyxy, conf, keypoints） 变成了 (xyxy, conf, cls， keypoints)  这一步应该不是必要操作，主要是yolov8源码中为了统一detect和pose的接口，通用nms后处理方法才这样做的
        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

pose export preds: torch.Size([1, 56, 6300])