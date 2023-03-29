# 3D hand reconstruction from a single image
[![wakatime](https://wakatime.com/badge/user/7974bf3e-99a6-4d26-8e4b-38ca6d5c9c64/project/ef5d8f38-163d-408b-8d57-ee7291b33fbf.svg)](https://wakatime.com/badge/user/7974bf3e-99a6-4d26-8e4b-38ca6d5c9c64/project/ef5d8f38-163d-408b-8d57-ee7291b33fbf)

This code is based on [S<sup>2</sup>HAND](https://github.com/TerenceCYJ/S2HAND).

S<sup>2</sup>HAND presents a self-supervised 3D hand reconstruction network that can jointly estimate pose, shape, texture, and the camera viewpoint. Specifically, we obtain geometric cues from the input image through easily accessible 2D detected keypoints. To learn an accurate hand reconstruction model from these noisy geometric cues, we utilize the consistency between 2D and 3D representations and propose a set of novel losses to rationalize outputs of the neural network. For the first time, we demonstrate the feasibility of training an accurate 3D hand reconstruction network without relying on manual annotations. For more details, please see our [paper](https://arxiv.org/abs/2103.11703), [video](https://youtu.be/tuQzu-UfSe8), and [project page](https://terencecyj.github.io/projects/CVPR2021/index.html).

## Code

### Environment (New)

cuda11.7 python3.9 pytorch1.13.
You may need to wake up your conda:
```sh
conda update -n base -c default conda
conda config --append channels conda-forge
conda update --all
```

```sh
conda env remove -n hand
conda create -n hand python=3.9
conda activate hand
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.7 -c pytorch -c nvidia

conda install -c fvcore -c iopath -c conda-forge fvcore iopath
# conda install pytorch3d -c pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

conda install tqdm tensorboard transforms3d scikit-image timm trimesh rtree opencv matplotlib
pip install chumpy
```

### Environment
Training is implemented with PyTorch. This code was developed under Python 3.6 and Pytorch 1.1.
I am using CUDA 10.2, so my environment is:
pytorch-1.6.0-py3.6_cuda10.2.89_cudnn7.6.5_0

```
conda create -n hand python=3.6
conda activate hand
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
```

Please compile the extension modules by running:
```
pip install tqdm tensorboardX transforms3d chumpy scikit-image timm

git clone https://github.com/TerenceCYJ/neural_renderer.git
cd neural_renderer
python setup.py install
rm -r neural_renderer
```
Note that we modified the ```neural_renderer/lighting.py``` compared to [daniilidis-group/neural_renderer](https://github.com/daniilidis-group/neural_renderer).

Note that under pytorch1.6, the neural_renderer may need to be modified before installation: https://github.com/facebookresearch/phosa/issues/6

For NIMBLE model, 
```sh
conda install trimesh rtree
conda install pytorch3d -c pytorch
```

### Data
For example, for 3D hand reconstruction task on the FreiHAND dataset:
- Download the FreiHAND dataset from the [website](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html).
- Modify the input and output directory accordingly in ```examples/config/FreiHAND/*.json```.

For HO3D dataset:
- Download the HO3D dataset from the [website](https://www.tugraz.at/index.php?id=40231).
- Modify the input and output directory accordingly in ```examples/config/HO3D/*.json```.

### Offline 2D Detection
- Offline 2D keypoint detection use a off-the-shelf detector like [pytorch-openpose](https://github.com/Hzzone/pytorch-openpose). 
   - We also provide detected 2D keypoints for [FreiHAND training set](https://www.dropbox.com/s/lx9nk8b90a2mgqy/freihand-train.json?dl=0). You may downlad and change the ```self.open_2dj_lists``` in the ```examples/data/dataset.py``` accordingly.
   - Or Download the ```hand_pose_model.pth``` provided by [pytorch-openpose](https://github.com/Hzzone/pytorch-openpose#download-the-models), and put the file to ```examples/openpose_detector/src```. Then use the following script  and modify the input and output directory accordingly. 
   (https://drive.google.com/file/d/1yVyIsOD32Mq28EHrVVlZbISDN7Icgaxw/view?usp=share_link)

        ```
        cd examples/openpose_detector
        python hand_detect.py```


### Training and Evaluation
#### HO3D
Evaluation: 
download the pretrained model [[texturehand_ho3d.t7]](https://www.dropbox.com/s/q5famyhzu19jv9o/texturehand_ho3d.t7?dl=0), and modify the ```"pretrain_model"``` in ```examples/config/HO3D/evaluation.json```.
```
cd S2HAND
python3 ./examples/train.py --config_json examples/config/HO3D/evaluation.json
```
Training:

Stage-wise training:
```
python3 ./examples/train.py --config_json examples/config/HO3D/SSL-shape.json
python3 ./examples/train.py --config_json examples/config/HO3D/SSL-kp.json
python3 ./examples/train.py --config_json examples/config/HO3D/SSL-finetune.json
```
Or end-to-end training:
```
python3 ./examples/train.py --config_json examples/config/HO3D/SSL-e2e.json
```
Note: remember to check and inplace the dirs and files in the ```*.json``` files.
#### FreiHAND
Evaluation: 
download the pretrained model [[texturehand_freihand.t7]](https://www.dropbox.com/s/kh4xxkfm08bh8py/texturehand_freihand.t7?dl=0), and modify the ```"pretrain_model"``` in ```examples/config/FreiHAND/evaluation.json```.
```
cd S2HAND
python3 ./examples/train.py --config_json examples/config/FreiHAND/evaluation.json
```
Training: refer to HO3D traing scripts.

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{chen2021s2hand,
    title={Model-based 3D Hand Reconstruction via Self-Supervised Learning}, 
    author={Chen, Yujin and Tu, Zhigang and Kang, Di and Bao, Linchao and Zhang, Ying and Zhe, Xuefei and Chen, Ruizhi and Yuan, Junsong},
    booktitle={Conference on Computer Vision and Pattern Recognition},
    year={2021}
}
```
