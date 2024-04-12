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
    x_diff = img_points - img_centre
    design_matrix = np.vstack([
        np.hstack([
            [x_diff[i, :] * world_points[i, 1]],
            [x_diff[i, :] * -world_points[i, 0]],
        ])
        for i in range(x_diff.shape[0])
    ])

    U, S, VT = np.linalg.svd(design_matrix.T)
    L_vec = VT[-1, :] / VT[-1, -1]

    return L_vec

def calculateTy(L_vec):
    return 1.0 / np.sqrt(np.power(L_vec[4:7], 2))

def calculateSx(L_vec, ty):
    return abs(ty) * np.sqrt(np.power(L_vec[0:3], 2))

def calculateRotation(L_vec, ty, sx):
    r1 = L_vec[0:3] * (ty / sx)
    r2 = L_vec[4:7] * ty
    R_mat = np.vstack([
        r1,
        r2,
        np.cross(r1, r2)
    ]) 
    return R_mat 

def calculateTx(L_vec, ty, sx):
    return L_vec[3] * (ty / sx)

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
    image_centres = np.zeros(image_points.shape)
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

        prompt = f"Calculate Rotation Matirx"
        printStart(prompt)
        R_mat = calculateRotation(L_vec, ty, sx)
        printEnd(prompt)

        prompt = f"Calculate tx"
        printStart(prompt)
        tx = calculateTx(L_vec, ty, sx)
        printEnd(prompt)

        prompt = f"Approximate f and tz"
        printStart(prompt)
        printEnd(prompt)

        prompt = f"Peforming non-linear optimization"
        printStart(prompt)
        printEnd(prompt)

        print("--------------------------------------------------------")



