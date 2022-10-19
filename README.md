# PoseIG Example of Deep High-Resolution Representation Learning for Human Pose Estimation

## Introduction
The utilization of PoseIG is based on the example from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch. Therefore, instruction to set up as well as the main programme is similar to the original repository. For illustration to use PoseIG, we use MSCOCO as the example.

## Quick start
### Installation
1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/).
   **Note that if you use pytorch's version < v1.0.0, you should following the instruction at <https://github.com/Microsoft/human-pose-estimation.pytorch> to disable cudnn's implementations of BatchNorm layer. We encourage you to use higher pytorch's version(>=v1.0.0)**
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools 
   ├── README.md
   └── requirements.txt
   ```

6. Download pretrained models from our model zoo([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
   ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            |-- pose_coco
            |   |-- pose_hrnet_w32_256x192.pth
            |   |-- pose_hrnet_w32_384x288.pth
            |   |-- pose_hrnet_w48_256x192.pth
            |   |-- pose_hrnet_w48_384x288.pth
            |   |-- pose_resnet_101_256x192.pth
            |   |-- pose_resnet_101_384x288.pth
            |   |-- pose_resnet_152_256x192.pth
            |   |-- pose_resnet_152_384x288.pth
            |   |-- pose_resnet_50_256x192.pth
            |   `-- pose_resnet_50_384x288.pth

   ```
   
### Data preparation

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 and test-dev2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

### PoseIG Package Illustration

#### Overview
All PoseIG related function is located at `lib/poseig_tools`. The function used to generate **poseig**, `epe.json` and `idx.json` is located at `lib/core/function`. For more information, you could refer to the documentation and comments that can be found in the script. 

#### Compute poseig and generate idx.json
To compute poseig for a certain dataset, you could modify the function 'lib/core/function:compute_poseig()'. In this package, the most important function is `lib/poseig_tools/ig.py:compute_poseig()`. To generate poseig for a certain model and dataset, you need to modify `back_func` and `back_info` in the function. For index computation, you could refer to `compute_DI()`, `compute_FI()`, `compute_LI()`, `compute_RI()` in `poseig_tools/index.py`.

Specifically, idx.json encodes a dictionary. Let's called it `idx_dict`. And each item in `idx_dict` encodes the index of a given sample. For instance, `idx_dict[2099]["DI"]` is an array (length is number of joints) containing DI of all joints of the 2099th sample in the dataset. To illustrate, `idx_dict` looks like this:

```
{'0': {
   'DI': [...],
   'FI': [...],
   'LI': [...].
   'target_weight': [...]
},
'1': {
   ...
},
...}
```

#### Compute epe and generate epe.json
To compute various indices, you could modify the function `lib/core/function:compute_epe()` and refer to `poseig_tools/index.py: compute_EPE()`. Similar to `idx.json`, it looks like:
```
{'0': {
   'EPE': [...],
   'target_weight': [...]
},
'1': {
   ...
},
...}
```

#### Save file
To generate index.json and epe.json, class `lib/poseig_tools/data_util:IG_DB` is used to track the output of all the algorithm. You could refer to `lib/core/function` to see how to use the class. Basically, it will create a folder called `IG_DB` in given path (a parameter used to create `IG_DB`). All PoseIG file will be store in `path/IG_DB/ig`. For example, the 9th sample in the dataset will be store as `9_ig.pickle` in it. File `epe.json` and index.json will be also saved in it.


### Testing on COCO val2017 dataset using model zoo's models([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))

#### Compute PoseIG

```
python tools/test.py --poseig --cfg experiments/coco/resnet/res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE models/pytorch/pose_coco/pose_resnet_50_256x192.pth TEST.USE_GT_BBOX True
```

#### Compute EPE

```
python tools/test.py --epe --cfg experiments/coco/resnet/res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE models/pytorch/pose_coco/pose_resnet_50_256x192.pth TEST.USE_GT_BBOX True
```

#### Visualize

```
python tools/test.py --vis --vis_sample 3633 --vis_joint 9 --cfg experiments/coco/resnet/res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE models/pytorch/pose_coco/pose_resnet_50_256x192.pth TEST.USE_GT_BBOX True
```

#### Check the output

All of the results will store in the output/coco/.../IG_DB, where the path is dependent on the model used for testing. For example, if we want to test model set by resnet/res50_256x192_d256x3_adam_lr1e-3.yaml, the output will be saved in output/coco/pose_resnet/PoseIG_human/output/coco/pose_resnet/res50_256x192_d256x3_adam_lr1e-3/IG_DB.

### Reference
We adopt code and model from the repository https://github.com/leoxiaobin/deep-high-resolution-net.

```
@inproceedings{sun2019deep,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={CVPR},
  year={2019}
}

@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}
```
