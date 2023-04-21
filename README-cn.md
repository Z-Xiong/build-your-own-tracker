# pytracking项目框架剖析及应用
pytracking是基于Pytorch的单目标跟踪跟踪框架，提供目标跟踪领域的公开数据集的前处理、数据的读入、模型快速搭建及模型训练。

# 项目模块和架构目录
## 主要模块
* pytracking -  for implementing your tracker
    * 通用跟踪和分割的数据集;
    * 跟踪性能和评价标准脚本;
    * 通用模块，包括深度网络，优化，特征抽取和相关滤波跟踪的使用。
* ltr - for training your tracker  
    * 训练数据集
    * 数据抽样和处理
    * 视觉跟踪网络模块

# 架构目录
* LTR：
    * admin: 用于加载模型，tensorboard，和环境设置，需要自己设置local.py
    * 数据相关 
        * dataset: 各种训练数据集的训练数据生成函数，也包含通过图像数据合成视频的脚本；
        * data_specs：数据集训练和验证的txt文件；
        * data：处理数据，如加载图像，数据增强，从视频中抽帧。
    * 网络相关 
        * external：训练额外依赖的包，这里有PreciseRoIPooling；
        * models：网络层定义。
    * 训练相关 
        * trainers:  训练器，输入actor，数据加载器，优化器，训练设置，以及学习率调节器，进行迭代训练；
        * actors : 用于计算loss， 输入数据（搜索图像，模板图像和搜索标注），输出训练loss；
        * train_settings: 使用run函数对各个跟踪算法的基本设置，包括训练数据集的选择，数据增强的选择，数据预处理，网络结构，loss函数，优化器，训练器设置。

```
ltr 
├── actors
│   ├── base_actor.py
│   ├── bbreg.py
│   ├── __init__.py
│   ├── __pycache__
│   ├── segmentation.py
│   └── tracking.py
├── admin
│   ├── environment.py
│   ├── __init__.py
│   ├── loading.py
│   ├── local.py  （定义训练日志存放路径和数据集路径）
│   ├── model_constructor.py
│   ├── multigpu.py
│   ├── __pycache__
│   ├── settings.py
│   ├── stats.py
│   └── tensorboard.py
├── checkpoints
├── data
│   ├── bounding_box_utils.py
│   ├── image_loader.py
│   ├── __init__.py
│   ├── loader.py
│   ├── processing.py （定义数据处理流程）
│   ├── processing_utils.py
│   ├── __pycache__
│   ├── sampler.py
│   └── transforms.py
├── dataset
│   ├── base_image_dataset.py
│   ├── base_video_dataset.py
│   ├── coco.py
│   ├── coco_seq.py
│   ├── davis.py
│   ├── ecssd.py
│   ├── got10k.py
│   ├── hku_is.py
│   ├── imagenetvid.py
│   ├── __init__.py
│   ├── lasot_candidate_matching.py
│   ├── lasot.py
│   ├── lvis.py
│   ├── msra10k.py
│   ├── __pycache__
│   ├── sbd.py
│   ├── synthetic_video_blend.py
│   ├── synthetic_video.py
│   ├── tracking_net.py
│   ├── vos_base.py
│   └── youtubevos.py
├── data_specs
│   ├── got10k_train_split.txt
│   ├── got10k_val_split.txt
│   ├── got10k_vot_exclude.txt
│   ├── got10k_vot_train_split.txt
│   ├── got10k_vot_val_split.txt
│   ├── lasot_train_split.txt
│   ├── lasot_train_train_split.txt
│   ├── lasot_train_val_split.txt
│   ├── trackingnet_classmap.txt
│   ├── youtubevos_jjtrain.txt
│   └── youtubevos_jjvalid.txt
├── external
│   └── PreciseRoIPooling
├── __init__.py
├── models
│   ├── backbone
│   ├── bbreg
│   ├── __init__.py
│   ├── kys
│   ├── layers
│   ├── loss
│   ├── lwl
│   ├── meta
│   ├── __pycache__
│   ├── target_candidate_matching
│   ├── target_classifier
│   └── tracking  （定义模型网络结构）
├── __pycache__
│   ├── __init__.cpython-37.pyc
│   ├── __init__.cpython-38.pyc
│   └── __init__.cpython-39.pyc
├── README.md
├── run_training.py
├── trainers
│   ├── base_trainer.py
│   ├── __init__.py
│   ├── ltr_trainer.py
│   └── __pycache__
└── train_settings
    ├── bbreg
    ├── dimp
    ├── __init__.py
    ├── keep_track
    ├── kys
    ├── lwl
    ├── your_track （使用run函数对所有模块的选择进行选择定义）
    └── __pycache__
```

# 如何训练自己的网络？
## 设置自定义的路径（训练日志以及数据集存放地址） 
### 在admin/local.py中设置如下 
```
class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '${PROJECT_PATH}/pytracking/workspace'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.lasot_dir = ''
        self.got10k_dir = '${DATA_PATH}/GOT-10k/train'
        self.trackingnet_dir = ''
        self.coco_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.lasot_candidate_matching_dataset_path = ''
```
* 在train_settings中创建your_track/YourTrack.py，并修改run函数中的自定义模块（这里的设置要基于以下Processing, Sampler, Network的设置）
* 设置Processing 
* 在data/processing.py中添加YourTrackProcessing类
* 设置Sampler 
* 在data/sampler.py中添加抽样方式，这里我们使用已经存在的ATOMSampler
* 设置Network 
* 在models/tracking下添加yourtrack.py，用于构造yourtrack跟踪网络