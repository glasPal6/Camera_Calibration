import numpy as np
import cv2
import os

from numpy.core.fromnumeric import shape

# DEBUG = True
DEBUG = False
def printStart(procName): print("[ ] - " + procName, end='\r')
def printEnd(procName): print("[x] - " + procName)

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
    lin_homg_pts = np.hstack([lin_homg_pts, np.ones((lin_homg_pts.shape[0], 1))])
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

def normalization_matrix(data):
    x, y = data[:, 0], data[:, 1]
    x_mean, y_mean = x.mean(), y.mean()
    x_var, y_var = x.var(), y.var()
    s_x, s_y = np.sqrt(2. / x_var), np.sqrt(2. / y_var)

    norm_matrix = np.array([[s_x,  0., -s_x * x_mean],
                            [ 0., s_y, -s_y * y_mean],
                            [ 0.,  0.,            1.]])
    return norm_matrix

def estimate_H_matrix(world_points, image_corners):
    assert(world_points.shape[0] >= 4)
    
    H_mats = []
    norm_world_matrix = normalization_matrix(world_points)
    norm_world = world_points @ norm_world_matrix.T
    for img_crn in image_corners:
        # Get the homography matrix for a single camera
        norm_img_crn_matrix = normalization_matrix(img_crn)
        norm_img_crn = np.hstack([img_crn, np.ones((img_crn.shape[0], 1))]) @ norm_img_crn_matrix.T
        design_matrix = np.vstack([
            np.vstack([
                np.hstack([norm_world[i],   np.zeros(3),    -norm_world[i]*norm_img_crn[i, 0]]),
                np.hstack([np.zeros(3),     norm_world[i],  -norm_world[i]*norm_img_crn[i, 1]]),
            ])
            for i in range(world_points.shape[0])
        ])

        U, S, VT = np.linalg.svd(design_matrix)
        h = VT[-1, :].reshape((3, 3))
        H_mats.append( np.linalg.inv(norm_img_crn_matrix) @ h @ norm_world_matrix)

    return np.array(H_mats)

def estimate_B_matrix(H_mats):
    # Contstuct the V matrix
    Vij = lambda hi, hj: np.array([ hi[0]*hj[0], hi[0]*hj[1] + hi[1]*hj[0], 
                    hi[1]*hj[1], 
                    hi[2]*hj[0] + hi[0]*hj[2], hi[2]*hj[1] + hi[1]*hj[2], 
                    hi[2]*hj[2] ]).T

    V = []
    for H in H_mats:
        v12 = Vij(H[:, 0], H[:, 1])
        v11 = Vij(H[:, 0], H[:, 0])
        v22 = Vij(H[:, 1], H[:, 1])
        V.append(v12.T)
        V.append((v11 - v22).T)
    V = np.array(V)

    # Solve and construct B
    U, S, VT = np.linalg.svd(V)
    b = VT[-1, :]

    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]],
    ])

    return B

def extract_K_int(B_mat):
    # Extract the K matrix from B
    try:
        L = np.linalg.cholesky(B_mat)
    except:
        try:
            L = np.linalg.cholesky(-B_mat)
        except:
            raise Exception("B Matrix is not positive definite")
    K_int = np.linalg.inv(L).T

    # Constust and constrain the K mat
    K_int /= K_int[2, 2]
    K_int[1, 0] = 0
    K_int[2, 0:2] = 0

    return K_int

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    SQUARE_SIZE = 12.5
    FOLDER_NAME = 'Calibration_Images_Zhang'
    SAVE_FOLDER = 'Save_Folder'
    HEIGHT, WIDTH = (7 - 1, 10 - 1) # -1 due to only taking the inner checkerboard

    # Extract the images
    printStart(f"Loading images from {FOLDER_NAME}")
    images = loadImages(FOLDER_NAME)
    if DEBUG: images = images[:1]
    printEnd("Loading images from {FOLDER_NAME}")

    printStart("Extracting Image and World Points")
    image_corners = getImagesPoints(images, HEIGHT, WIDTH)
    world_corners = getWorldPoints(SQUARE_SIZE, HEIGHT, WIDTH)
    assert(image_corners.shape[1] == world_corners.shape[0])
    if DEBUG: displayCorners(images, image_corners, HEIGHT, WIDTH, SAVE_FOLDER)
    printEnd("Extracting Image and World Points")

    printStart("Estimating the Homography Matricies")
    H_mats = estimate_H_matrix(world_corners, image_corners)
    printEnd("Estimating the Homography Matricies")

    printStart("Estimating the B Matricies")
    B_mat = estimate_B_matrix(H_mats)
    printEnd("Estimating the B Matricies")
    
    printStart("Estimating the Internal Matrix")
    K_int = extract_K_int(B_mat)
    print(K_int)
    printEnd("Estimating the Internal Matrix")
