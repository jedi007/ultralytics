https://github.com/ultralytics/ultralytics 官方教程：https://docs.ultralytics.com/modes/train/

资源安装

更建议下载代码后使用 下面指令安装，这样可以更改源码，如果不需要更改源码就直接pip install ultralytics也是可以的。

需要到ultralytics目录下执行以下指令
pip install -e .

这样安装后，可以直接修改yolov8源码，并且可以立即生效。此图是命令解释： 安装成功后： pip list可以看到安装的包：





detect

export pred shape: shape(1,84,6300)  :  (batch_size, [xywh + numclasses], numboxes)
train pred shape: 
[torch.Size([1, 144, 80, 80]), torch.Size([1, 144, 40, 40]), torch.Size([1, 144, 20, 20])]  三个尺寸boxes下的输出
转化为： torch.Size([1, 64, 8400]) , torch.Size([1, 80, 8400]) 
前者是（batch_szize, 16*4, 80*80+40*40+20*20） 8400是3尺度合一的boxes数量,  16*4 是4个坐标数据：(x1,y1)(x2,y2), 16是格子数，按照分类的思想，x1预测从0-15的偏移，然后加权就和做为其预测值
后者是 (batch_size, num_classes, 80*80+40*40+20*20)



pose:
pose train preds : ([torch.Size([1, 65, 80, 80]), torch.Size([1, 65, 40, 40]), torch.Size([1, 65, 20, 20])],  torch.Size([1, 51, 8400]))
        #  torch.Size([1, 65, 80, 80])  : (batchsize, reg_max * 4 + num_classes, grid, grid)
        #  torch.Size([1, 51, 8400]):  (batchsize, keypoints(17*3), 80*80+40*40+20*20)

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