import numpy as np


class OffsetDepthDhruv(object):
	def __init__(self):
		pass

	def __call__(self, data):
		rgbA, depthA,poseA = data
		depthA = self.normalize_depth(depthA, poseA)
		return rgbA.astype(np.float32), depthA, poseA

	def normalize_depth(self, depth, pose):
		depth = depth.astype(np.float32)
		invalid_mask = np.logical_or(depth<=100, depth>=2000)
		if pose[2, 3]<0:  #gl pose
			depth += pose[2, 3] * 1000
		else:
			depth -= pose[2, 3] * 1000

		depth[invalid_mask] = 2000
		assert (depth<=2000).all()
		return depth



class NormalizeChannels(object):
	def __init__(self,mean,std):
		self.mean = mean
		self.std = std


	def __call__(self, data):
		rgbA, depthA, rgbB, depthB, maskA, maskB, poseA = data
		rgbA, depthA = self.normalize_channels(rgbA, depthA, self.mean[:4], self.std[:4])
		rgbB, depthB = self.normalize_channels(rgbB, depthB, self.mean[4:], self.std[4:])
		return rgbA, depthA, rgbB, depthB, maskA, maskB,poseA

	def normalize_channels(self, rgb, depth, mean, std):
		rgb = rgb.transpose(2,0,1)
		rgb = (rgb-mean[:3, np.newaxis, np.newaxis])/std[:3, np.newaxis, np.newaxis]
		depth = (depth-mean[3, np.newaxis, np.newaxis])/std[3, np.newaxis, np.newaxis]
		return rgb, depth


class NormalizeChannelsRender(object):
	def __init__(self,mean,std):
		self.mean = mean
		self.std = std


	def __call__(self, data):
		rgbA, depthA, poseA = data
		rgbA, depthA = self.normalize_channels(rgbA, depthA, self.mean[:4], self.std[:4])
		return rgbA, depthA, poseA

	def normalize_channels(self, rgb, depth, mean, std):
		rgb = rgb.transpose(2,0,1)
		rgb = (rgb-mean[:3, np.newaxis, np.newaxis])/std[:3, np.newaxis, np.newaxis]
		depth = (depth-mean[3, np.newaxis, np.newaxis])/std[3, np.newaxis, np.newaxis]
		return rgb, depth


class NormalizeChannelsReal(object):
	def __init__(self,mean,std):
		self.mean = mean
		self.std = std
	def __call__(self, data):
		rgbA, depthA, poseA = data
		rgbA, depthA = self.normalize_channels(rgbA, depthA, self.mean[4:], self.std[4:])
		return rgbA, depthA, poseA
	def normalize_channels(self, rgb, depth, mean, std):
		rgb = rgb.transpose(2,0,1)
		rgb = (rgb-mean[:3, np.newaxis, np.newaxis])/std[:3, np.newaxis, np.newaxis]
		depth = (depth-mean[3, np.newaxis, np.newaxis])/std[3, np.newaxis, np.newaxis]
		return rgb, depth