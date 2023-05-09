# 应该在界面启动的时候就将模型加载出来，设置tmp的目录来放中间的处理结果
import shutil
import PyQt5.QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import threading            # 用于多线程编程
import argparse             # 用于解析命令行参数
import os
import sys
from pathlib import Path    # 用于操作文件和目录路径
import cv2
import torch
import torch.backends.cudnn as cudnn
import os.path as osp

FILE = Path(__file__).resolve()  # 获取当前文件的绝对路径
ROOT = FILE.parents[0]  # 获取当前文件所在目录的父目录，即YOLOv5的根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将YOLOv5的根目录添加到系统路径中，以便后续的模块导入
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 将YOLOv5的根目录转换为相对路径

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

# 添加一个关于界面
# 窗口主类
class MainWindow(QTabWidget):
    # 定义了一个名为MainWindow的类，该类继承自QTabWidget，用于创建一个带有多个选项卡的主窗口。
    # class MainWindow(QTabWidget): ：定义了一个名为MainWindow的类，该类继承自QTabWidget。
    # QTabWidget：是Qt中的一个类，用于创建带有多个选项卡的窗口。
    # 该类可以添加多个选项卡，每个选项卡可以包含不同的控件，例如按钮、文本框、标签等。您可以在该类中添加各种控件和布局，以实现自己的界面设计

    def __init__(self):
        # 初始化界面
        super().__init__()  # 调用父类QTabWidget的构造函数，初始化窗口对象
        self.setWindowTitle('Target detection system')  # 设置窗口标题
        self.resize(1200, 800)  # 设置窗口大小
        self.setWindowIcon(QIcon("images/UI/lufei.png"))  # 设置窗口图标

        # 图片读取进程
        self.output_size = 480  # 设置输出图像的大小
        self.img2predict = ""  # 初始化要进行预测的图像路径
        self.device = 'cpu'  # 设置使用的设备（CPU或GPU）

        # 初始化视频读取线程
        self.vid_source = '0'  # 设置视频源，初始设置为摄像头
        self.stopEvent = threading.Event()  # 创建一个线程事件对象
        '''
        threading.Event()是Python标准库threading中的一个类，用于创建线程事件对象。
        线程事件对象可以用于线程同步，它提供了一个标志，用于线程之间的通信。
        在这里，self.stopEvent是一个线程事件对象，用于控制视频读取线程的运行状态。
        在程序运行过程中，如果需要停止视频读取线程，可以调用stopEvent.set()方法将线程事件标志设置为True，从而通知线程停止运行。
        因此，重置线程事件对象的初始状态是非常重要的，可以通过stopEvent.clear()方法将线程事件标志设置为False，从而确保线程能够正常运行
        '''
        self.webcam = True  # 初始设置为使用摄像头
        self.stopEvent.clear()  # 清除线程事件
        self.model = self.model_load(weights="runs/train/exp_yolov5s/weights/best.pt", device=self.device)  # 加载预训练的YOLOv5模型

        self.initUI()  # 调用initUI函数，用于创建用户界面
        self.reset_vid()  # 重置视频读取线程

    '''
    ***模型初始化***
    '''
    @torch.no_grad()  # 关闭PyTorch的自动求导机制
    def model_load(self, weights="", device='', half=False, dnn=False):
        device = select_device(device)  # 选择使用的设备（CPU或GPU），并返回一个PyTorch设备对象
        half &= device.type != 'cpu'  # 如果使用的设备不是CPU，则将half标志设置为True
        device = select_device(device)  # 再次选择使用的设备（确保设备的正确性）
        model = DetectMultiBackend(weights, device=device, dnn=dnn)  # 创建一个DetectMultiBackend对象，用于加载YOLOv5模型
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        # 获取模型的步长、类别名称、模型权重文件路径、是否使用JIT编译、是否使用ONNX等信息

        half &= pt and device.type != 'cpu'  # 如果使用的设备不是CPU且模型权重文件存在，则将half标志设置为True
        if pt:
            model.model.half() if half else model.model.float()  # 将模型转换为半精度或全精度

        print("模型加载完成!")
        return model  # 返回加载好的YOLOv5模型

    '''
    ***界面初始化***
    '''
    def initUI(self):
        font_title = QFont('楷体', 16)  # 标题字体
        font_main = QFont('楷体', 14)  # 正文字体

        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()
        # 创建了一个空白的图形界面窗口，我们可以在这个窗口中添加其他的控件
        img_detection_layout = QVBoxLayout()  # 垂直布局
        img_detection_title = QLabel("图片识别功能")  # 标题
        img_detection_title.setFont(font_title)     # 设置控件的字体
        mid_img_widget = QWidget()  # 中间的图片区域
        mid_img_layout = QHBoxLayout()  # 水平布局
        self.left_img = QLabel()  # 左边的图片
        self.right_img = QLabel()  # 右边的图片
        self.left_img.setPixmap(QPixmap("images/UI/up.jpeg"))  # 设置默认图片
        self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))  # 设置默认图片
        self.left_img.setAlignment(Qt.AlignCenter)  # 居中显示
        self.right_img.setAlignment(Qt.AlignCenter)  # 居中显示
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("上传图片")  # 上传图片按钮
        det_img_button = QPushButton("开始检测")  # 开始检测按钮
        up_img_button.clicked.connect(self.upload_img)  # 绑定上传图片函数
        det_img_button.clicked.connect(self.detect_img)  # 绑定检测图片函数
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")  # 设置按钮样式
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")  # 设置按钮样式
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)  # 添加标题
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)  # 添加图片区域
        img_detection_layout.addWidget(up_img_button)  # 添加上传图片按钮
        img_detection_layout.addWidget(det_img_button)  # 添加开始检测按钮
        img_detection_widget.setLayout(img_detection_layout)  # 设置布局

        # 视频检测子界面
        vid_detection_widget = QWidget()
        vid_detection_layout = QVBoxLayout()  # 垂直布局
        vid_title = QLabel("视频检测功能")  # 标题
        vid_title.setFont(font_title)
        self.vid_img = QLabel()  # 视频区域
        self.vid_img.setPixmap(QPixmap("images/UI/up.jpeg"))  # 设置默认图片
        vid_title.setAlignment(Qt.AlignCenter)  # 标题居中显示
        self.vid_img.setAlignment(Qt.AlignCenter)  # 视频区域居中显示
        self.webcam_detection_btn = QPushButton("摄像头实时监测")  # 摄像头实时监测按钮
        self.mp4_detection_btn = QPushButton("视频文件检测")  # 视频文件检测按钮
        self.vid_stop_btn = QPushButton("停止检测")  # 停止检测按钮
        self.webcam_detection_btn.setFont(font_main)
        self.mp4_detection_btn.setFont(font_main)
        self.vid_stop_btn.setFont(font_main)
        self.webcam_detection_btn.setStyleSheet("QPushButton{color:white}"
                                                 "QPushButton:hover{background-color: rgb(2,110,180);}"
                                                 "QPushButton{background-color:rgb(48,124,208)}"
                                                 "QPushButton{border:2px}"
                                                 "QPushButton{border-radius:5px}"
                                                 "QPushButton{padding:5px 5px}"
                                                 "QPushButton{margin:5px 5px}")  # 设置按钮样式
        self.mp4_detection_btn.setStyleSheet("QPushButton{color:white}"
                                              "QPushButton:hover{background-color: rgb(2,110,180);}"
                                              "QPushButton{background-color:rgb(48,124,208)}"
                                              "QPushButton{border:2px}"
                                              "QPushButton{border-radius:5px}"
                                              "QPushButton{padding:5px 5px}"
                                              "QPushButton{margin:5px 5px}")  # 设置按钮样式
        self.vid_stop_btn.setStyleSheet("QPushButton{color:white}"
                                         "QPushButton:hover{background-color: rgb(2,110,180);}"
                                         "QPushButton{background-color:rgb(48,124,208)}"
                                         "QPushButton{border:2px}"
                                         "QPushButton{border-radius:5px}"
                                         "QPushButton{padding:5px 5px}"
                                         "QPushButton{margin:5px 5px}")  # 设置按钮样式
        self.webcam_detection_btn.clicked.connect(self.open_cam)  # 绑定打开摄像头函数
        self.mp4_detection_btn.clicked.connect(self.open_mp4)  # 绑定打开视频文件函数
        self.vid_stop_btn.clicked.connect(self.close_vid)  # 绑定关闭视频函数
        # 添加组件到布局上
        vid_detection_layout.addWidget(vid_title)  # 添加标题
        vid_detection_layout.addWidget(self.vid_img)  # 添加视频区域
        vid_detection_layout.addWidget(self.webcam_detection_btn)  # 添加摄像头实时监测按钮
        vid_detection_layout.addWidget(self.mp4_detection_btn)  # 添加视频文件检测按钮
        vid_detection_layout.addWidget(self.vid_stop_btn)  # 添加停止检测按钮
        vid_detection_widget.setLayout(vid_detection_layout)  # 设置布局

        # todo 关于界面
        # 创建一个 QWidget 对象，该对象将用作该界面的主窗口
        about_widget = QWidget()
        # 创建一个 QVBoxLayout 对象，该对象将用作 about_widget 对象的布局管理器
        about_layout = QVBoxLayout()
        # 创建一个 QLabel 对象，该对象是一个标签，用于显示欢迎词语
        about_title = QLabel('欢迎使用目标检测系统\n\n 提供付费指导：有需要的好兄弟加下面的QQ即可')
        # 设置 about_title 的字体为 "楷体"，大小为 18
        about_title.setFont(QFont('楷体', 18))
        # 将 about_title 对齐方式设置为居中
        about_title.setAlignment(Qt.AlignCenter)
        # 创建一个 QLabel 对象，该对象是一个标签，用于显示 QQ 图标
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/UI/qq.png'))
        # 将 about_img 对齐方式设置为居中
        about_img.setAlignment(Qt.AlignCenter)

        # label4.setText("<a href='https://oi.wiki/wiki/学习率的调整'>如何调整学习率</a>")
        # 创建一个 QLabel 对象，该对象是一个标签，用于显示作者信息和链接
        label_super = QLabel()
        # 设置 label_super 的文本和链接
        label_super.setText("<a href='https://blog.csdn.net/ECHOSON'>或者你可以在这里找到我-->肆十二</a>")
        # 设置 label_super 的字体为 "楷体"，大小为 16
        label_super.setFont(QFont('楷体', 16))
        # 设置 label_super 打开外部链接的功能
        label_super.setOpenExternalLinks(True)
        # 将 label_super 对齐方式设置为右对齐
        label_super.setAlignment(Qt.AlignRight)

        # 将 about_title、about_img 和 label_super 添加到 about_layout 布局管理器中
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_layout.addWidget(label_super)
        # 将 about_layout 设置为 about_widget 的布局管理器
        about_widget.setLayout(about_layout)

        # 将 self.left_img 对象的对齐方式设置为居中
        self.left_img.setAlignment(Qt.AlignCenter)
        # 将三个标签页添加到主窗口中，分别是 img_detection_widget、vid_detection_widget 和 about_widget
        self.addTab(img_detection_widget, '图片检测')
        self.addTab(vid_detection_widget, '视频检测')
        self.addTab(about_widget, '联系我')
        # 将三个标签页的图标设置为 "images/UI/lufei.png"
        self.setTabIcon(0, QIcon('images/UI/lufei.png'))
        self.setTabIcon(1, QIcon('images/UI/lufei.png'))
        self.setTabIcon(2, QIcon('images/UI/lufei.png'))

    '''
     ***上传图片***
     '''
    def upload_img(self):
        # 选择图片文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        '''
        创建一个文件对话框，用于选择要打开的文件
        self：表示当前窗口的对象，用于指定文件对话框的父窗口，这里传入 self 表示文件对话框是当前窗口的子窗口。
        'Choose file'：表示文件对话框的标题，用于提示用户选择的文件类型。
        ''：表示文件对话框的起始路径，这里传入空字符串表示从当前目录开始选择文件。
        '*.jpg *.png *.tif *.jpeg'：表示文件对话框能够选择的文件类型，这里限定了可以选择的文件类型为 .jpg、.png、.tif、.jpeg。
        '''

        if fileName:
            # 获取文件的后缀名
            suffix = fileName.split(".")[-1]
            # 构造保存路径，将图片复制到指定目录
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)

            # 读取图片，调整图片大小并保存
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            '''
            im0：表示要调整大小的图片。
            (0, 0)：表示输出图片的大小，这里传入 (0, 0) 表示输出大小与输入大小按比例缩放。
            fx=resize_scale, fy=resize_scale：表示缩放比例，这里使用了上面计算得到的 resize_scale
            '''
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)

            # 将上传的图片路径存储到 self.img2predict 中
            self.img2predict = fileName

            # 将处理后的图片显示在左侧的 QLabel 控件中
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))

            # 重置右侧的 QLabel 控件显示为默认占位图片
            self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))

    '''
    ***检测图片***
    '''         # 待
    def detect_img(self):
        model = self.model  # 加载模型
        output_size = self.output_size  # 输出图像的大小
        source = self.img2predict  # 输入的图片来源，可以是文件/目录/URL/通配符，0表示摄像头
        imgsz = [640, 640]  # 推理时的图像大小（像素）
        conf_thres = 0.25  # 置信度阈值
        iou_thres = 0.45  # NMS的IOU阈值
        max_det = 1000  # 每张图片的最大检测数量
        device = self.device  # CUDA设备，比如0或0,1,2,3或cpu
        view_img = False  # 是否显示结果
        save_txt = False  # 是否将结果保存为*.txt
        save_conf = False  # 是否在--save-txt标签中保存置信度
        save_crop = False  # 是否保存裁剪后的预测框
        nosave = False  # 是否不保存图像/视频
        classes = None  # 按类别过滤：--class 0 或 --class 0 2 3
        agnostic_nms = False  # 类别不可知的NMS
        augment = False  # 增强推理
        visualize = False  # 可视化特征
        line_thickness = 3  # 边框线宽（像素）
        hide_labels = False  # 隐藏标签
        hide_conf = False  # 隐藏置信度
        half = False  # 使用FP16半精度推理
        dnn = False  # 使用OpenCV DNN进行ONNX推理
        print(source)
        if source == "":
            QMessageBox.warning(self, "请上传", "请先上传图片再进行检测")
        else:
            source = str(source)
            device = select_device(self.device)
            webcam = False
            stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
            imgsz = check_img_size(imgsz, s=stride)  # 检查图像大小
            save_img = not nosave and not source.endswith('.txt')  # 保存推理图像
            # Dataloader
            if webcam:
                view_img = check_imshow()
                cudnn.benchmark = True  # 设为True可加速恒定图像大小推理
                dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
                bs = len(dataset)  # batch_size
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
                bs = 1  # batch_size
            vid_path, vid_writer = [None] * bs, [None] * bs
            # 进行推理
            if pt and device.type != 'cpu':
                model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # 预热
            dt, seen = [0.0, 0.0, 0.0], 0
            for path, im, im0s, vid_cap, s in dataset:
                t1 = time_sync()
                im = torch.from_numpy(im).to(device)
                im = im.half() if half else im.float()  # uint8转为fp16/32
                im /= 255  # 0-255转为0.0-1.0
                if len(im.shape) == 3:
                    im = im[None]  # 扩展batch维度
                t2 = time_sync()
                dt[0] += t2 - t1
                # 推理
                # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)
                t3 = time_sync()
                dt[1] += t3 - t2
                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                dt[2] += time_sync() - t3
                # 二阶段分类器（可选）
                # pred = utils.general.apply_classifier(pred, modelc, img, im0s)

                # 处理检测结果
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)  # 将路径转为 Path 类型
                    s += '%gx%g ' % im.shape[2:]  # 打印图片尺寸
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化增益
                    imc = im0.copy() if save_crop else im0  # 用于保存裁剪图像
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))  # 可视化注释器

                    if len(det):
                        # 将检测框从 img_size 转换为 im0 的大小
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                        # 打印检测结果
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # 统计每个类别的检测数量
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 添加到字符串中

                        # 可视化结果
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # 写入文件
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 归一化的 xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 标签格式
                                # with open(txt_path + '.txt', 'a') as f:
                                #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or save_crop or view_img:  # 添加边界框到图像
                                c = int(cls)  # 类别
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                                # if save_crop:
                                #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                                #                  BGR=True)

                    # 打印时间（仅推断）
                    LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

                    # 可视化结果
                    im0 = annotator.result()
                    # if view_img:
                    #     cv2.imshow(str(p), im0)
                    #     cv2.waitKey(1)  # 1 毫秒

                    # 保存结果（带有检测结果的图像）
                    resize_scale = output_size / im0.shape[0]
                    im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
                    cv2.imwrite("images/tmp/single_result.jpg", im0)
                    self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))

    # 视频检测，逻辑基本一致，有两个功能，分别是检测摄像头的功能和检测视频文件的功能，先做检测摄像头的功能。

    '''
    ### 界面关闭事件 ### 
    '''
    def closeEvent(self, event):
        reply = QMessageBox.question(self,  # 窗口对象
                                     'quit',  # 标题
                                     "Are you sure?",  # 消息框中显示的文本
                                     QMessageBox.Yes | QMessageBox.No,  # 显示的按钮类型
                                     QMessageBox.No)  # 默认按钮
        if reply == QMessageBox.Yes:  # 如果用户点击了 Yes 按钮
            self.close()  # 关闭窗口
            event.accept()  # 接受事件
        else:  # 如果用户点击了 No 按钮
            event.ignore()  # 忽略事件，继续运行程序

    '''
    ### 视频关闭事件 ### 
    '''
    def open_cam(self):
        self.webcam_detection_btn.setEnabled(False)  # 禁用“摄像头检测”按钮
        self.mp4_detection_btn.setEnabled(False)  # 禁用“视频检测”按钮
        self.vid_stop_btn.setEnabled(True)  # 启用“停止检测”按钮
        self.vid_source = '0'  # 摄像头的 ID
        self.webcam = True  # 标记为使用摄像头
        th = threading.Thread(target=self.detect_vid)  # 创建一个线程用于检测
        th.start()  # 启动线程开始检测

    '''
    ### 开启视频文件检测事件 ### 
    '''
    def open_mp4(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')  # 打开文件对话框，选择要打开的视频文件
        if fileName:  # 如果选择了文件
            self.webcam_detection_btn.setEnabled(False)  # 禁用“摄像头检测”按钮
            self.mp4_detection_btn.setEnabled(False)  # 禁用“视频检测”按钮
            # self.vid_stop_btn.setEnabled(True)
            self.vid_source = fileName  # 视频文件路径
            self.webcam = False  # 标记为使用视频文件
            th = threading.Thread(target=self.detect_vid)  # 创建一个线程用于检测
            th.start()  # 启动线程开始检测

    '''
    ### 视频开启事件 ### 
    '''
    # 视频和摄像头的主函数是一样的，不过是传入的source不同罢了
    def detect_vid(self):
        # 设置参数
        model = self.model
        output_size = self.output_size
        imgsz = [640, 640]  # 推理时的图像大小
        conf_thres = 0.25  # 置信度阈值
        iou_thres = 0.45  # NMS IOU 阈值
        max_det = 1000  # 每张图像的最大检测框数
        view_img = False  # 是否显示结果图像
        save_txt = False  # 是否保存检测结果到 txt 文件
        save_conf = False  # 是否保存检测结果置信度到标注文件
        save_crop = False  # 是否保存检测框裁剪出的图像
        nosave = False  # 是否不保存结果图像或视频
        classes = None  # 检测特定类别的物体，默认为 None 表示检测所有类别
        agnostic_nms = False  # 是否使用类别不可知的 NMS
        augment = False  # 是否使用数据增强
        visualize = False  # 是否可视化特征图
        line_thickness = 3  # 检测框线条粗细
        hide_labels = False  # 是否隐藏检测结果标签
        hide_conf = False  # 是否隐藏检测结果置信度
        half = False  # 是否使用 FP16 低精度模式
        dnn = False  # 是否使用 OpenCV DNN 进行 ONNX 推理
        source = str(self.vid_source)  # 视频源路径
        webcam = self.webcam  # 是否使用摄像头
        device = select_device(self.device)  # 选择设备进行推理
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(imgsz, s=stride)  # 检查图像尺寸是否合法
        save_img = not nosave and not source.endswith('.txt')  # 是否保存推理结果图像
        # 数据加载器
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # 设置为 True 可以加速推理速度
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs
        # 进行推理
        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # 预热模型
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # 转换数据类型，从 uint8 到 fp16/32
            im /= 255  # 将像素值从 0-255 转换为 0.0-1.0
            if len(im.shape) == 3:
                im = im[None]  # 如果没有批次维度则添加批次维度
            t2 = time_sync()
            dt[0] += t2 - t1
            # 进行推理
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2
            # 进行 NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3
            # 处理预测结果
            for i, det in enumerate(pred):  # 每张图像
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # 转换为 Path 对象

                s += '%gx%g ' % im.shape[2:]
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # 将检测框从 img_size 转换到 im0 大小
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # 打印结果
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # 每个类别的检测框数
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                    # 保存结果
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # 将结果写入 txt 文件
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                -1).tolist()  # 物体在图像中的归一化坐标
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 标注的格式
                            # with open(txt_path + '.txt', 'a') as f:
                            #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # 将检测结果画在图像上
                            c = int(cls)  # 类别索引
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))

                            # if save_crop:  # 裁剪检测框区域并保存为图像文件
                            #     save_one_box(xyxy, imc, file=str(p.with_suffix(f'.jpg')), BGR=True)

                        # Print time (inference-only)
                        # 打印推理时间
                        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

                        # Stream results
                        # 将结果实时流式传输
                        im0 = annotator.result()
                        frame = im0
                        resize_scale = output_size / frame.shape[0]
                        frame_resized = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                        # 将结果图像保存并显示在界面上
                        cv2.imwrite("images/tmp/single_result_vid.jpg", frame_resized)
                        self.vid_img.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))

                        if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                            # 如果用户点击了停止按钮则退出循环
                            self.stopEvent.clear()
                            self.webcam_detection_btn.setEnabled(True)
                            self.mp4_detection_btn.setEnabled(True)
                            self.reset_vid()
                            break
                        # self.reset_vid()  # 重置视频窗口

    '''
    ### 界面重置事件 ### 
    '''
    def reset_vid(self):
        # 将摄像头检测按钮和视频检测按钮的状态设置为启用
        self.webcam_detection_btn.setEnabled(True)
        self.mp4_detection_btn.setEnabled(True)
        # 将视频窗口的图像设置为 "images/UI/up.jpeg"
        self.vid_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        # 将视频源设置为默认值 '0'，表示使用摄像头作为视频源
        self.vid_source = '0'
        # 将 webcam 变量设置为 True，表示使用摄像头进行检测
        self.webcam = True

    '''
    ### 视频重置事件 ### 
    '''
    def close_vid(self):
        # 设置 stopEvent 标志为 True，表示停止视频检测
        self.stopEvent.set()
        # 调用 reset_vid 方法，将视频窗口重置为初始状态
        self.reset_vid()

if __name__ == "__main__":
    # 创建一个 QApplication 对象，并将命令行参数传递给它
    app = QApplication(sys.argv)
    # 创建一个 MainWindow 对象
    mainWindow = MainWindow()
    # 显示主窗口
    mainWindow.show()
    # 进入事件循环，并等待应用程序退出
    sys.exit(app.exec_())