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
- Don't make a loader for training yet. Make sure the pipeline works on the pretrained data and finish the bayesian framework this week. This includes the code modularity (1 day)
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

- Tracknet is weird - gives liealgebra with dt multiplied (in a way)
So we dont get angular velocities, we get relative pose