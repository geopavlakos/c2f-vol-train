# Coarse-to-Fine Volumetric Prediction for Single-Image 3D Human Pose (Training code)
## Georgios Pavlakos, Xiaowei Zhou, Konstantinos G. Derpanis, Kostas Daniilidis

This is the training code for the paper **Coarse-to-Fine Volumetric Prediction for Single-Image 3D Human Pose**. Please follow the links to read the [paper](https://arxiv.org/abs/1611.07828) and visit the corresponding [project page](https://www.seas.upenn.edu/~pavlakos/projects/volumetric). This code follows closely the [original training code for the Stacked Hourglass networks](https://github.com/anewell/pose-hg-train) by Alejandro Newell, so you can follow the corresponding release for an elaborate description on the command line arguments and options. Here, we provide details so that you can train a network with a volumetric output for 3D human pose estimation (or generally 3D keypoint localization).

For the testng code please visit this [repository](https://github.com/geopavlakos/c2f-vol-demo).

We provide code and data to train our models on [Human3.6M](http://vision.imar.ro/human3.6m/description.php). Please follow the instructions below to setup and use our code. To run this code, make sure the following are installed:

- [Torch7](https://github.com/torch/torch7)
- hdf5
- cudnn

### 1) Data format

We provide the data for the training and testing set of Human3.6M. Please run the following script to get all the relevant data (**be careful, since the size is over 32GB**)

```
data.sh
```

These images are extracted from the videos of the [original dataset](http://vision.imar.ro/human3.6m/description.php), and correspond to the images used for testing by the most typical protocol. The filename protocol we follow is:

```
S[subject number]_[Action Name].[Camera Name]_[Frame Number].jpg
```

An example for Subject 5, performing action Eating (iteration 1), when we consider camera name '55011271' and frame 321, is:

```
S5_Eating_1.55011271_000321.jpg
```

Check also the files:

```
data/hm36m/annot/train_images.txt
data/hm36m/annot/valid_images.txt
```

to figure out the complete list of training and testing images.

### 2) Training

You can train a typical Coarse-to-Fine Volumetric prediction model by using the command line:

```
th main.lua -dataset h36m -expID test-run-c2f -netType hg-stacked-4 -task pose-c2f -nStack 4 -resZ 1,2,4,64 -LR 2.5e-4 -nEpochs 1000 -trainIters 1000 -validIters 1000
```

Please check the file ''opts.lua'' for all the relevant command line options. Our code follows closely the [original training code for the Stacked Hourglass networks](https://github.com/anewell/pose-hg-train) by Alejandro Newell, so you can follow the corresponding repository for an elaborate description on the command line arguments and options. An additional argument required to model our class of networks is 'resZ'. This is a list with the resolution of the z-dimension for each hourglass' output. The length of the list must match the number of the hourglass components ''nStack''.

Also, to replicate the models used in our paper, please use the architectures defined in the files:

```
src/models/hg-stacked-2.lua
src/models/hg-stacked-3.lua
src/models/hg-stacked-4.lua
```

with 2,3 and 4 hourglasses respectively. Again, these follow the original Stacked Hourglass network design. Alternatively, you can use the typical hourglass architecture:

```
src/models/hg.lua
```

which has a more uniform network design. If you are using a single hourglass (no iterative coarse-to-fine prediction), you can train a simple model by using the command line:

```
th main.lua -dataset h36m -expID test-run-vol -netType hg -task pose-vol -nStack 1 -resZ 64 -LR 2.5e-4 -nEpochs 1000 -trainIters 1000 -validIters 1000
```

### 3) Evaluation

You can evaluate your trained model on users S9 and S11 of Human3.6M, by running:

```
th main.lua -dataset h36m -expID test-run-c2f -task pose-c2f -nStack 4 -finalPredictions 1 -nEpochs 0 -validIters 109867 -loadModel \path\to\model
```

or you can use our [demo code](https://github.com/geopavlakos/c2f-vol-demo) for that.

### 4) Training on your own data

Compared to training a hourglass with a 2D output, the only overhead our code requires is to provide the index for the z-Dimension (zind), for each keypoint. We provide this in a 1-64 scale, and the code adapts it when the resolution is smaller. As long as you provide this information during training, along with the pixel locations of each keypoint, you should be able to use our training code on your custom data.

### Citing

If you find this code useful for your research, please consider citing the following paper:

	@Inproceedings{pavlakos17volumetric,
	  Title          = {Coarse-to-Fine Volumetric Prediction for Single-Image 3{D} Human Pose},
	  Author         = {Pavlakos, Georgios and Zhou, Xiaowei and Derpanis, Konstantinos G and Daniilidis, Kostas},
	  Booktitle      = {Computer Vision and Pattern Recognition (CVPR)},
	  Year           = {2017}
	}

### Acknowledgements

This code follows closely the [released code](https://github.com/anewell/pose-hg-train) for the Stacked Hourglass networks by Alejandro Newell. We gratefully appreciate the impact it had on our work. If you use our code, please consider citing the [original paper](http://arxiv.org/abs/1603.06937) as well.
