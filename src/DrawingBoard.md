# Pipeline
## Depth Completition

Depth images are in mm and the maximum value is 3.2 meters. 
Parameters: 
- Max Depth is kept as 5 meters
- Diamond Kernal of size 7 is used
- Kernel5, kernal7 and *kernel17* is used instead of kernal31. 

The reason for choosing kernal17 was that it was less distorting the depth map (although it didnt matter on the object)


## Dataset
There are 2 modes - test and train. Depending on the mode, the folder structures are different since testing is not on synthetic data but training is on that. The way I am going to do this is by specifying the "synthetic and not syntheic" flag rather than the train/test flag. The reason is that I might also want to train on real data. 

dataset_info considers data about size of the dataset, .ply file, intrinsics and crop resolution parameter. 

### Data Augmentations
Depths are normalized, images are normalized. These are non trivial here, and needs to be handled carefully in the dataset. For a unit testing scenario, a custom test pipeline needs to be created with openGL rendering before starting to train. 
Transformations are done after cropping. Cropping depends on the pose estimation. Thus, we can't possibly transform our data. This is how it is done in the tracknet. However this is not nescessay (uint8 hinderence), I need to specifically test for that case.  


## Checklist 
- ~~Wait for the author to clarify on Frames in dataloader.~~ No response from author
- ~~Next steps are data augmentation tests, and inference replication of the pretrained model.~~
#### 11
Inference replicated with the functions from orignal code with horrible depth map. Noticeable improvement with depth completion with better kernels
- Don't make a loader for training yet. ~~Make sure the pipeline works on the pretrained data and finish the bayesian framework this week. ~~ This includes the code modularity (1 day)
- Synthetic Data dataloader.  (1 day)
#### 18
At this point, I will be taking a backseat.
- Developing the model and training it
#### 25
- Tie with the ICP results and wrap up the project




### Pipeline
Currently replicating the tracknet inference code, but build on it to make a state based framework


## Possible Bugs
Paper transforms the data after cropping. I am assuming in rgb its fine, but teh normalize depth function might cause problem(?) if the order is reversed. - Lot more errors than expected on this part. 


## Error State Kalman filter

- Propogation using SE3Tracknet 
    - Update Error State Covariance
    - Update Nominal State
- Measurment Update step from the ICP
    - Update error state covariance
    - Calculate Kalman Gain
    - Update Nominal State



## 14th Nov Updates
- Still can't figure out a way to make the dataloader cleaner. Note that we can't alter data before cropping. This is because, the cropping works based on 3D pose, we backproject 3D points with pose and crop using that. The current solution would be to load raw images, and let a utils class handle both cropping and augmentation. Our architecture will assume that dataloader's job is to only provide raw images. 

- This architecture simplies the problem but requires utils to work on batches!


- Turns out synthetic data is already cropped which make the training pipeline extremely simple!


- Just because I can, doesn't mean I would. Due to interest of time, we will switch to lightning and metrics. Sorry! 


## Metrics
- ADD, MSE, LR, Gradient Norm, Weight Norm, 

## Parameters
- lr, btch, loss weights,

We will use early stopping for sure


## 15th Nov updates
- We donot need data augmentation since we already have domain randomized images. Depth data however is perfect in simulation. We can in theory using Dropout to simulate depth image holes. 
--> I trained a ruin Depth network that simulates noisy depth data from synthetic image. Since this is offline refinement, this is fine to me. The network is also just 5 layers deep and performs segmentation on the real and bad data. 

## 18th Nov Updates
- Training decisions
RGB normalizations of mean and std are weird. Training on the raw synthetic data for now and see the network performance. Then we will use ruin depth to train and see that model performance.

- Training pipeline works. As of today, I will complete the metrics part. 
-- Loss, 

## 19th Nov Update:
New depth network. I just realized I was not using max pooling operations. So we were just using CNNs without downsampling which makes less sense. Anyways, we get the same performance. Training pipeline is complete. I am now bored with the project since I got better results. 