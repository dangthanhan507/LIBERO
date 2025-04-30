from numpy import ndarray
import open3d as o3d
import numpy as np

def get_rgbd_image(rgb_img: ndarray, depth_img: ndarray) -> o3d.geometry.RGBDImage:
    rgb_img, depth_img = o3d.geometry.Image(rgb_img), o3d.geometry.Image(depth_img)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img, depth_img)
    return rgbd_image

def get_point_cloud(rgbd_image, intrinsic_matrix: ndarray, camera_height: int, camera_width: int) -> o3d.geometry.PointCloud:
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(camera_height, camera_width, intrinsic_matrix)
    )
    return point_cloud

def get_colored_point_cloud(point_cloud: o3d.geometry.PointCloud, rgb_image: ndarray) -> ndarray:
    point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    colored_points = np.concatenate((np.asarray(point_cloud.points), np.asarray(rgb_image).reshape(-1, 3)), axis=1)
    return colored_points