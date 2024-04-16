import numpy as np
from scipy.optimize import curve_fit
import cv2
from glob import glob

DEBUG = True
# DEBUG = False
debugPrint = print if DEBUG else lambda *a, **k: None
def printStart(procName): debugPrint("[ ] - " + procName, end='\r')
def printEnd(procName): debugPrint("[x] - " + procName)
#    prompt = f""
#    printStart(prompt)
#    printEnd(prompt)

import warnings
warnings.filterwarnings("ignore", message="Covariance of the parameters could not be estimated")

#-------------------------------------------------------------------------------

def loadData(folder_name):
    """
    Load all the images in the given folder
    """
    world_points_file = f"{folder_name}/world_points.txt"
    image_files = sorted(glob(f"{folder_name}/*.jpg"), key=lambda x: int(x[len(folder_name) + 1]))
    image_points_file = sorted(glob(f"{folder_name}/*_image_points.txt"), key=lambda x: int(x[len(folder_name) + 1]))

    world_points = np.loadtxt(world_points_file)
    world_points = np.hstack([world_points, np.ones((world_points.shape[0], 1))])
    images = []
    image_points = []
    for img, pt in zip(image_files, image_points_file):
        images.append(cv2.imread(img))
        image_points.append(np.loadtxt(pt))

    return world_points, np.array(image_points), np.array(images)

#-------------------------------------------------------------------------------

def calculateLvec(world_points, img_points, img_centre):
    measurements = img_points - img_centre
    design_matrix = np.vstack([
        np.hstack([
             world_points[i, :] * measurements[i, 1],
             world_points[i, :] * -measurements[i, 0],
        ])
        for i in range(measurements.shape[0])
    ])

    U, S, VT = np.linalg.svd(design_matrix)
    L_vec = VT[-1, :] / VT[-1, -1]

    return L_vec

def calculateTy(L_vec):
    return 1.0 / np.linalg.norm(L_vec[4:7])

def calculateSx(L_vec, ty):
    return abs(ty) * np.linalg.norm(L_vec[0:3])

def calculateTx(L_vec, ty, sx):
    return L_vec[3] * (ty / sx)

def calculateRotation(L_vec, ty, sx):
    r1 = L_vec[0:3] * (ty / sx)
    r2 = L_vec[4:7] * ty
    # r1_t = L_vec[0:3] * (ty / sx)
    # r2_t = L_vec[4:7] * ty
    # k = -0.5 * r1_t @ r2_t
    # r1 = r1_t + k * r2_t
    # r2 = r2_t + k * r1_t
    # r1 /= np.linalg.norm(r1)
    # r2 /= np.linalg.norm(r2)
    R_mat = np.vstack([
        r1,
        r2,
        np.cross(r1, r2)
    ]) 
    return R_mat 


def calculateFandTz(world_points, image_points, R_mat, ty, tx, sx):
    A = np.vstack([
        np.vstack([
            np.hstack([(sx * R_mat[0, :] @ world_points[i, 0:3] + tx), 0, -image_points[i, 0]]),
            np.hstack([0, (sx * R_mat[1, :] @ world_points[i, 0:3] + ty), -image_points[i, 1]]),
        ])
        for i in range(world_points.shape[0])
    ])
    b = np.vstack([
        np.vstack([
            R_mat[2, :] @ world_points[i, 0:3] * image_points[i, 0],
            R_mat[2, :] @ world_points[i, 0:3] * image_points[i, 1],
        ])
        for i in range(world_points.shape[0])
    ])

    x = np.linalg.pinv(A) @ b

    return x[0, 0], x[1, 0], x[2, 0]

def calculate_radial_distortion(world_points, image_points, K_int, E_ext):
    u_c = np.array([K_int[0,2], K_int[1,2]])

    # Observed distortion error and Model distortion error
    d_dot = []
    D = []

    # Projected sensor points
    u_proj = reproject_pin_hole(world_points.T, K_int, E_ext)

    d_dot.append(u_proj.T - image_points)

    r = np.linalg.norm(u_proj[:2, :].T, axis=1).reshape((-1, 1))
    D.append(
        np.hstack([
            np.vstack((u_proj[:2, :].T - u_c) * np.power(r, 2)),
            np.vstack((u_proj[:2, :].T - u_c) * np.power(r, 4)),
        ])
    )

    d_dot = np.array(d_dot).reshape((-1, 1))
    D = np.array(D).reshape((-1, 2))

    k_rad = np.linalg.pinv(D) @ d_dot

    return k_rad.reshape((2))

#-------------------------------------------------------------------------------

def pack_parameters(K_int, E_ext, k_rad):
    packed_params = []

    alpha, beta, gamma, u_c, v_c = K_int[0,0], K_int[1,1], K_int[0,1], K_int[0,2], K_int[1,2]
    k0, k1 = k_rad
    packed_params.extend([alpha, beta, gamma, u_c, v_c, k0, k1])

    R = E_ext[:3, :3]
    t = E_ext[:, 3]
    rodrigues = cv2.Rodrigues(R)[0]
    packed_params.extend(rodrigues.reshape((3)))
    packed_params.extend(t)

    return np.array(packed_params)

def unpack_parameters(parameters):
    alpha, beta, gamma, u_c, v_c, k0, k1 = parameters[:7]
    K_int = np.array([[alpha, gamma, u_c],
                  [   0.,  beta, v_c],
                  [   0.,    0.,  1.]])
    k_rad = np.array([k0, k1])

    rho_x, rho_y, rho_z, t_x, t_y, t_z = parameters[7:7+6]
    R = cv2.Rodrigues(np.array([rho_x, rho_y, rho_z]))[0]
    t = np.array([t_x, t_y, t_z])
    E_ext = np.hstack([R, t[:, np.newaxis]])

    return K_int, E_ext, k_rad

def parameter_refinement_loss_func(world_points, *params):
    K_int, E_ext, k_rad = unpack_parameters(params)

    u_proj = reproject_radial_distortion(world_points.T, K_int, E_ext, k_rad)
    img_points = u_proj[:2, :].T

    return img_points.flatten()

def parameter_refinement(world_points, image_points, K_int, E_ext, k_rad):
    param0 = pack_parameters(K_int, E_ext, k_rad)

    popt, pcov = curve_fit(parameter_refinement_loss_func, world_points, image_points.flatten(), param0)
    
    K_int, E_ext, k_rad = unpack_parameters(popt)
    return K_int, E_ext, k_rad

#-------------------------------------------------------------------------------

def reproject_pin_hole(world_points, K_int, E_ext):
    P = K_int @ E_ext
    u_proj = P @ world_points
    u_proj /= u_proj[-1]
    return u_proj[:2, :]

def reproject_radial_distortion(world_points, K_int, E_ext, k_rad):
    camera_points = E_ext @ world_points
    camera_points = camera_points[:2, :] / camera_points[2, :]

    r = np.linalg.norm(camera_points, axis=0).reshape((1, -1))
    distortion_warping = 1 + k_rad[0] * np.power(r, 2) + k_rad[1] * np.power(r, 4)
    distorted_points = distortion_warping * camera_points

    projected_points = K_int[:2, :] @ np.vstack([distorted_points, np.ones((1, distorted_points.shape[1]))])

    return projected_points

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    SQUARE_SIZE = 12.5
    FOLDER_NAME = 'Calibration_Images_Tsai'
    SAVE_FOLDER = 'Save_Folder'
    HEIGHT, WIDTH = (7 - 1, 10 - 1) # -1 due to only taking the inner checkerboard

    # Extract the images
    prompt = f"Loading Data from {FOLDER_NAME}"
    printStart(prompt)
    world_points, image_points, images = loadData(FOLDER_NAME)
    image_centres = np.zeros((image_points.shape[0], 2))
    assert(world_points.shape[0] == image_points.shape[1])
    printEnd(prompt)

    for c, img, img_points, img_centre in zip(range(image_points.shape[0]), images, image_points, image_centres):
        print("--------------------------------------------------------")
        print(f"Calculating parameters for Camera {c + 1}")
        print("--------------------------------------------------------")

        prompt = f"Calculate L vector"
        printStart(prompt)
        L_vec = calculateLvec(world_points, img_points, img_centre)
        printEnd(prompt)

        prompt = f"Calculate ty"
        printStart(prompt)
        ty = calculateTy(L_vec)
        printEnd(prompt)
        print("\nNeed to check the sign of ty\n")

        prompt = f"Calculate s"
        printStart(prompt)
        sx = calculateSx(L_vec, ty)
        printEnd(prompt)

        prompt = f"Calculate tx"
        printStart(prompt)
        tx = calculateTx(L_vec, ty, sx)
        printEnd(prompt)

        prompt = f"Calculate Rotation Matirx"
        printStart(prompt)
        R_mat = calculateRotation(L_vec, ty, sx)
        printEnd(prompt)

        prompt = f"Approximate f and tz"
        printStart(prompt)
        fx, fy, tz = calculateFandTz(world_points, img_points, R_mat, ty, tx, sx)
        printEnd(prompt)
        E_ext = np.hstack([R_mat, np.vstack([tx, ty, tz])])
        K_int = np.array([
            [fx, sx, img_centre[0]],
            [0, fy, img_centre[1]],
            [0, 0, 1]
        ])
        print("Intrinsic Parameters:")
        print(K_int)
        print("Extrinsic Parameters:")
        print(E_ext)

        prompt = "Addind the Z axis to the world coordinates"
        printStart(prompt)
        world_points = np.hstack([
            world_points[:, :2], 
            np.zeros((world_points.shape[0], 1)), 
            np.ones((world_points.shape[0], 1))
        ])
        printEnd(prompt)

        prompt = "Estimating the Radial Distortion"
        printStart(prompt)
        k_rad = calculate_radial_distortion(world_points, img_points, K_int, E_ext)
        printEnd(prompt)
        print("Radial Distortion:")
        print(f"\tk1 - {k_rad[0]}")
        print(f"\tk2 - {k_rad[1]}")

        prompt = "Preforming non-linear optimization"
        printStart(prompt)
        K_int, E_ext, k_rad = parameter_refinement(world_points, img_points, K_int, E_ext, k_rad)
        printEnd(prompt)
        print("Intrinsic Parameters:")
        print(K_int)
        print("Extrinsic Parameters:")
        print(E_ext)
        print("Radial Distortion:")
        print(f"\tk1 - {k_rad[0]}")
        print(f"\tk2 - {k_rad[1]}")

        print("--------------------------------------------------------")



