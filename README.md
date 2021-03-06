#####################################################################################
</br>
<h1><b>U-Net: Convolutional Networks for Biomedical Image Segmentation</b></h1>

<h3>The link for the U-Net paper</h3> https://arxiv.org/pdf/1505.04597.pdf
<h4>U-Net is an architecture for semantic segmentation.
It consists of a contracting path and an expansive path. 
The contracting path follows the typical architecture of a convolutional network. 
It consists of the repeated application of two 3x3 convolutions (unpadded convolutions), 
each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling.
At each downsampling step we double the number of feature channels.
Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution 
(“up-convolution”) that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path,
and two 3x3 convolutions, each followed by a ReLU.
The cropping is necessary due to the loss of border pixels in every convolution.
At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes.
In total the network has 23 convolutional layers.</h4>
</br>
<img src="https://raw.githubusercontent.com/zhixuhao/unet/master/img/u-net-architecture.png" alt="U-Net">
</br>

 #####################################################################################
 
 <h1><b>Road Ectraction by Deep Residual U-Net</b></h1>
 
<h3>The link for the U-Net paper</h3> https://arxiv.org/pdf/1711.10684.pdf
ResUNet, a semantic segmentation model inspired by the deep residual learning and UNet. An architecture that take advantages from both(Residual and UNet) models.
</br>
<img src="https://raw.githubusercontent.com/nikhilroxtomar/Deep-Residual-Unet/master/images/arch.png" alt="Residual_U-Net">


 
