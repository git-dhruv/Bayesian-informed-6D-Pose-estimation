import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation 

class lieGroup:
    """
    Helper for Lie Groups and Lie Algebra
    """
    def __init__(self):
        pass
    
    @staticmethod
    def constructValidRotationMatrix(R):
        """Finds the least square estimate of closed SO(3)"""
        U,_,Vt = np.linalg.svd(R)
        Smod = np.eye(3)
        Smod[-1,-1] = np.linalg.det(U@Vt)
        return U@Smod@Vt

    
    def makeHomoTransform(self, t, q):
        R = Rotation.from_quat(q).as_matrix()
        H = np.column_stack((R, t.reshape(-1,1)))
        H = np.row_stack((H,np.array([[0,0,0,1]])))
        return H

    def dissolveHomoTransform(self, H):
        q = Rotation.from_matrix(H[:3,:3]).as_quat()
        t = H[:3,-1]
        return t,q
    
    



class inputImageHandler:
    """
    Class invokes the cropping
    """
    def __init__(self, outputsize, objectWidth, K):
       self.objectWidth = objectWidth
       self.outputsize = outputsize
       self.K = K
    def compute_bbox(self, pose, K, width=230, scale=(1, 1, 1)):
        x = pose[0, 3] * scale[0]
        y = pose[1, 3] * scale[1]
        z = pose[2, 3] * scale[2]
        offset = width / 2

        points = np.ndarray((4, 3), dtype=np.float)
        points[0] = [x - offset, y - offset, z]     # top left
        points[1] = [x - offset, y + offset, z]     # top right
        points[2] = [x + offset, y - offset, z]     # bottom left
        points[3] = [x + offset, y + offset, z]     # bottom right

        #Backprojecting 3D points onto the image
        projected_vus = np.zeros((points.shape[0], 2))
        projected_vus[:, 1] = points[:, 0] * K[0,0] / points[:, 2] + K[0,2]
        projected_vus[:, 0] = points[:, 1] * K[1,1] / points[:, 2] + K[1,2]
        projected_vus = np.round(projected_vus).astype(np.int32)
        return projected_vus



    def crop_bbox(self,color, depth, boundingbox, output_size=(100, 100)):
        left = np.min(boundingbox[:, 1])
        right = np.max(boundingbox[:, 1])
        top = np.min(boundingbox[:, 0])
        bottom = np.max(boundingbox[:, 0])

        if len(color.shape)>3:
            raise Exception ("Function not compatible with batch operations or rgbd image!")

        h, w, c = color.shape
        crop_w = right - left
        crop_h = bottom - top
        color_crop = np.zeros((crop_h, crop_w, 3), dtype=color.dtype)
        depth_crop = np.zeros((crop_h, crop_w), dtype=np.float)
        top_offset = abs(min(top, 0))
        bottom_offset = min(crop_h - (bottom - h), crop_h)
        right_offset = min(crop_w - (right - w), crop_w)
        left_offset = abs(min(left, 0))

        top = max(top, 0)
        left = max(left, 0)
        bottom = min(bottom, h)
        right = min(right, w)
        color_crop[top_offset:bottom_offset, left_offset:right_offset, :] = color[top:bottom, left:right, :]
        depth_crop[top_offset:bottom_offset, left_offset:right_offset] = depth[top:bottom, left:right]
        resized_rgb = cv2.resize(color_crop, output_size, interpolation=cv2.INTER_NEAREST)
        resized_depth = cv2.resize(depth_crop, output_size, interpolation=cv2.INTER_NEAREST)
        
        mask_rgb = resized_rgb != 0
        mask_depth = resized_depth != 0
        resized_depth = resized_depth.astype(np.uint16)
        final_rgb = resized_rgb * mask_rgb
        final_depth = resized_depth * mask_depth        
        return final_rgb, final_depth
    
    def cropImage(self, rgb, depth, pose, scale=(1000,1000,1000)):
        """
        Function assumes that the rgb is batched (with size 1) and is tensor
        """
        rgb = rgb[0].cpu().numpy(); depth = depth[0].cpu().numpy()
        bb = self.compute_bbox(pose, self.K, self.objectWidth, scale)
        return self.crop_bbox(rgb, depth, bb, self.outputsize)



class pointCloudUtils:
    def __init__(self):
        pass
    def toOpen3dCloud(self, points, colors=None):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is not None:
            cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64)/255.0)
        return cloud

class imageOps:
    def __init__(self):
        pass

    def toHomogenous(self, pts):
        return np.hstack((pts, np.ones((pts.shape[0], 1))))
    
    def bckPrjctFromK(self, K, pts):
        homogeneous_points = self.toHomogenous(pts)
        projected_points = np.dot(K@np.eye(4)[:3,:], homogeneous_points.T).T
        projected_points[:, 0] /= projected_points[:, 2]
        projected_points[:, 1] /= projected_points[:, 2]
        projected_points = np.round(projected_points[:, :2]).astype(np.int32)
        return projected_points


    def bckPrjctFromP(self, K, pose, pts):
        # pts is 4XN or 3XN
        if pts.shape[0] != 4:
            pts = self.toHomogenous(pts)        
        return K@(pose@pts)

