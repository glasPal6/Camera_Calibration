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
    assert(world_points.shape[0] == image_points.shape[1])
    printEnd(prompt)

    for i, img, img_points in zip(range(image_points.shape[0]), images, image_points):
        print("--------------------------------------------------------")
        print(f"Calculating parameters for Camera {i + 1}")
        print("--------------------------------------------------------")

        prompt = f"Calculate L vector"
        printStart(prompt)
        printEnd(prompt)

        prompt = f"Calculate ty"
        printStart(prompt)
        printEnd(prompt)

        prompt = f"Calculate s"
        printStart(prompt)
        printEnd(prompt)

        prompt = f"Calculate Rotation Matirx"
        printStart(prompt)
        printEnd(prompt)

        prompt = f"Calculate tx"
        printStart(prompt)
        printEnd(prompt)

        prompt = f"Approximate f and tz"
        printStart(prompt)
        printEnd(prompt)

        prompt = f"Peforming non-linear optimization"
        printStart(prompt)
        printEnd(prompt)

        print("--------------------------------------------------------")



