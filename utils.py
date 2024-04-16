import numpy as np

#-------------------------------------------------------------------------------

def reproject_pin_hole(world_points, K_int, E_ext):
    P = K_int @ E_ext
    u_proj = P @ world_points
    u_proj /= u_proj[-1]
    return u_proj[:2, :]

def reproject_radial_distortion(world_points, K_int, E_ext, k_rad):
    camera_points = (E_ext @ world_points)
    camera_points = camera_points[:2, :] / camera_points[2, :]

    r = np.linalg.norm(camera_points, axis=0).reshape((1, -1))
    distortion_warping = 1 + k_rad[0] * np.power(r, 2) + k_rad[1] * np.power(r, 4)
    distorted_points = distortion_warping * camera_points

    projected_points = K_int[:2, :] @ np.vstack([distorted_points, np.ones((1, distorted_points.shape[1]))])

    return projected_points

#-------------------------------------------------------------------------------



