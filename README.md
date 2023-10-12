# CT-Inverse-Problem
Computed Tomography is a diagnostic imaging technology that uses x-rays to measure the projection of an object from all directions, and from that data reconstructs the linear attenuation coefficient throughout the object. Filtered back projection (FBP) has been the standard CT image reconstruction method for 4 decades. A simple, fast, and reliable technique, FBP has delivered high-quality images in several clinical applications. However, with faster and more advanced CT scanners, FBP has become increasingly obsolete. Higher image noise and more artifacts are especially noticeable in lower-dose CT imaging using FBP. This performance gap was partly addressed by model-based iterative reconstruction (MBIR). However its long reconstruction time has limited widespread application. Hybrid iterative reconstruction partially addressed these limitations by blending FBP with MBIR and is currently the state-of-the-art reconstruction technique. The clinical introduction of deep learning reconstruction  algorithms over the past 5 years has demonstrated the potential of deep learning to rapidly reconstruct images with low noise, desired noise texture, and preserved spatial resolution while providing the opportunity to reduce radiation dose.   

**This is a research project under the guidance of Dr C.S. Sastry at Indian Institute of Technology, Hyderabad. We investigate the potential and performance of deep learning algorithms to reconstruct CT Images and observe if they can outperform the current traditional state of the art techniques.**

The Data that I used consists of 4000 Image pairs, where each pair has True Breast Phantom image and its 128-view FBP reconstructed counterpart. ResUNet is the first Architecture that we used.

## **1. RESUNET:**

### 1.1 Architecture
#### 1.1.1 Basic Buidling Block

 Residual units consists of:
 * Two 3x3 convolutional block.
 * A Skip Connection from the input to the output of the residual block.

 Convolutional block consists of:
 * One Batch Normalization layer.
 * One ReLU Activation layer.
 * One Convolutional layer.
   
 ![Residual Block](Convolutional_block_RESUNET.png)
 Figure(b) is the residual block.

 #### 1.1.2 Entire Architecture

 
