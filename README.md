# MRI_SR_3DGS
Supervised 3D Gaussian Splatting-based MRI super-resolution:

Using the scheme of "Pretrained 3DGS" and 3D LR as input to predict the intensity and 3D Gaussian distribution of HR matrtix.
The HR matrix has the same resolution as the 3D HR image. The image is reconstructed using the weighted sum of the predicted intensities, 
and the weight is defined as the pdf of the Gaussian distributions that covers the corresponding voxel.
The reconstructed HR image will be compared to ground truth to calculate losses and update the parameters of the network.
