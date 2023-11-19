# Meta Tracker

- ~~Pipeline test on ground truth~~
- ~~Depth Completion~~
- Classical 6D ICP with covariance
- ~~Network architecture~~
- Logging
- ~~Data (Dataloader, Augmentations and OpenGL rendering)~~
- Profiling
- Ablation Studies (Depth Completion Ablations are done)

# Documentation

## Running the original SE3Tracknet  
Ofcourse, before setting up, we need to see if the model is working. Our project focuses on Mustard bottle, since that is given by real data as well. 
### Setting up the code
```
git clone https://github.com/wenbowen123/iros20-6d-pose-tracking
```
Note that gitignore will take care of not pushing this onto our codebase. Some non trivial requirements are as follows:
```
pip install click
pip install vispy
pip install plyfile
```
Rest are opencv/numpy stuff. If you encounter openCV error on windows, reinstalling it worked for me. 
Ref: https://stackoverflow.com/questions/67120450/error-2unspecified-error-the-function-is-not-implemented-rebuild-the-libra
### Data to download
We are using mustard data. Download the following:
YCB Video Data: https://archive.cs.rutgers.edu/archive/a/2020/pracsys/Bowen/iros2020/YCB_Video_data_organized/0050.tar.gz
Ply Files: https://archive.cs.rutgers.edu/pracsys/se3_tracknet/YCB_models_with_ply.zip
Synthetic Training Data: https://archive.cs.rutgers.edu/archive/a/2020/pracsys/Bowen/iros2020/YCB_traindata/mustard_bottle.tar.gz
Weights: Download Pretrained Weights on YCB Video data.

### Running the code:
```
cd iros20-6d-pose-tracking
python predict.py --mode ycbv --ycb_dir C:\Users\dhruv\Desktop\680Final\data\ --seq_id 50 --train_data_path C:\Users\dhruv\Desktop\680Final\data\mustard_bottle\train_data_blender_DR --ckpt_dir C:\Users\dhruv\Desktop\680Final\weights\YCB_weights\mustard_bottle\model_epoch150.pth.tar --mean_std_path C:\Users\dhruv\Desktop\680Final\weights\YCB_weights\mustard_bottle --class_id 5 --model_path C:\Users\dhruv\Desktop\680Final\data\CADmodels\006_mustard_bottle\textured.ply --outdir C:\Users\dhruv\Desktop\680Final\logs
```
Don't change class id. 


# WARNING!!
Current code is extremely unstable and would probably not run in any system except mine (I haven't even uploaded some files for a start). 