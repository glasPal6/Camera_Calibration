import numpy as np
import cv2
import os

def loadImages(folder_name):
    files = os.listdir(folder_name)
    print("Loading images from ", folder_name)
    images = []
    for f in files:
        # print(f)
        image_path = folder_name + "/" + f
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)			
        else:
            print("Error in loading image ", image)

    return images

def getImagesPoints(imgs, h, w):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    images = imgs.copy()
    all_corners = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            corners2 = corners2.reshape(-1,2)
            # corners2 = np.hstack((corners2.reshape(-1,2), np.ones((corners2.shape[0], 1))))
            all_corners.append(corners2)
    return all_corners

def getWorldPoints(square_side, h, w):
    # h, w = [6, 9]
    Yi, Xi = np.indices((h, w)) 
    offset = 0
    lin_homg_pts = np.stack(((Xi.ravel() + offset) * square_side, (Yi.ravel() + offset) * square_side)).T
    return lin_homg_pts

if __name__ == "__main__":
    SQUARE_SIZE = 12.5
    FOLDER_NAME = 'Calibration_Images_Zhang'

    # Extract the images
    images = loadImages(FOLDER_NAME)
    h, w = [6,9]
    all_image_corners = getImagesPoints(images, h, w)
    world_corners = getWorldPoints(SQUARE_SIZE, h, w)
