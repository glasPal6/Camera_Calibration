import numpy as np
import cv2
import os

DEBUG = True

#-------------------------------------------------------------------------------

def loadImages(folder_name):
    """
    Load all the images in the given folder
    """
    files = os.listdir(folder_name)
    images = []
    for f in files:
        # print(f)
        image_path = folder_name + "/" + f
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)			
        else:
            print("Error in loading image ", image)
            raise FileNotFoundError
    
    return np.array(images)

def getImagesPoints(imgs, h, w):
    """
    Find the chessboard corners in the images
    """
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
            all_corners.append(corners2)
        else:
            raise ValueError("Corners not Found")

    return np.array(all_corners)

def getWorldPoints(square_side, h, w):
    Yi, Xi = np.indices((h, w)) 
    offset = 0
    lin_homg_pts = np.stack(((Xi.ravel() + offset) * square_side, (Yi.ravel() + offset) * square_side)).T
    return lin_homg_pts

def displayCorners(images, all_corners, h, w, save_folder):
    for i, image in enumerate(images):
        corners = all_corners[i]
        corners = np.float32(corners.reshape(-1, 1, 2))
        
        cv2.drawChessboardCorners(image, (w, h), corners, True)
        img = cv2.resize(image, (int(image.shape[1]/3), int(image.shape[0]/3)))

        filename = save_folder + '/IMG_' + str(i) + ".png"
        cv2.imwrite(filename, img)
        
        if DEBUG:
            cv2.imshow('frame', img)
            cv2.waitKey()

    if DEBUG:
        cv2.destroyAllWindows()

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    SQUARE_SIZE = 12.5
    FOLDER_NAME = 'Calibration_Images_Zhang'
    SAVE_FOLDER = 'Save_Folder'
    HEIGHT, WIDTH = (6,9)

    # Extract the images
    print(f"[ ] - Loading images from {FOLDER_NAME}", end='\r')
    images = loadImages(FOLDER_NAME)
    print(f"[x] - Loading images from {FOLDER_NAME}")

    print("[ ] - Extracting Image and World Points", end='\r')
    all_image_corners = getImagesPoints(images, HEIGHT, WIDTH)
    world_corners = getWorldPoints(SQUARE_SIZE, HEIGHT, WIDTH)
    assert(all_image_corners.shape[1:] == world_corners.shape)
    if DEBUG: displayCorners(images, all_image_corners, HEIGHT, WIDTH, SAVE_FOLDER)
    print("[x] - Extracting Image and World Points")
