import numpy as np
from scipy.optimize import curve_fit
import cv2
import os


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

def H_opt_f_refine(world_points, *params):
    """
    Value function for Levenberg-Marquardt refinement.
    """
    h0, h1, h2, h3, h4, h5, h6, h7, h8 = params
    
    H = np.array([
        [h0, h1, h2],
        [h3, h4, h5],
        [h6, h7, h8],
    ])

    img_points = H @ world_points.T
    img_points /= img_points[-1]

    return img_points[:2].T.flatten()

def H_opt_jac_refine(world_points, *params):
    """
    Jacobian function for Levenberg-Marquardt refinement.
    """
    h0, h1, h2, h3, h4, h5, h6, h7, h8 = params
    
    s_x = h0 * world_points[:, 0] + h1 * world_points[:, 1] + h2
    s_y = h3 * world_points[:, 0] + h4 * world_points[:, 1] + h5
    w   = h6 * world_points[:, 0] + h7 * world_points[:, 1] + h8
    w_sq = w**2

    J = np.vstack([
        np.vstack([
            np.hstack([ world_points[i] / w[i], np.zeros(3),            (-s_x[i] * world_points[i]) / w_sq[i]]),
            np.hstack([ np.zeros(3),            world_points[i] / w[i], (-s_y[i] * world_points[i]) / w_sq[i]]),
        ])
        for i in range(world_points.shape[0])
    ])

    return np.array(J)

def estimate_H_matrix(world_points, image_corners):
    assert(world_points.shape[0] >= 4)
    
    H_mats = []
    norm_world_matrix = normalization_matrix(world_points)
    norm_world = world_points @ norm_world_matrix.T
    for img_crn in image_corners:
        # Setup the design matrix to estimate H
        norm_img_crn_matrix = normalization_matrix(img_crn)
        norm_img_crn = np.hstack([img_crn, np.ones((img_crn.shape[0], 1))]) @ norm_img_crn_matrix.T
        design_matrix = np.vstack([
            np.vstack([
                np.hstack([norm_world[i],   np.zeros(3),    -norm_world[i]*norm_img_crn[i, 0]]),
                np.hstack([np.zeros(3),     norm_world[i],  -norm_world[i]*norm_img_crn[i, 1]]),
            ])
            for i in range(world_points.shape[0])
        ])

        # Estimate H 
        U, S, VT = np.linalg.svd(design_matrix)
        h = VT[-1, :].reshape((3, 3))
        h = np.linalg.inv(norm_img_crn_matrix) @ h @ norm_world_matrix

        # Preform non-linear estimation of H
        popt, pcov = curve_fit(H_opt_f_refine, world_points, img_crn.flatten(), p0=h.flatten(), jac=H_opt_jac_refine)
        h_refined = popt.reshape((3, 3))

        H_mats.append(h_refined / h_refined[2, 2])

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

def extract_R_t_ext(H_mat, K_int):
    h0, h1, h2 = H_mat[:,0], H_mat[:,1], H_mat[:,2]
    K_inv = np.linalg.inv(K_int)

    lambda_ = 1. / np.linalg.norm(np.dot(K_inv, h0))
    r0 = lambda_ * np.dot(K_inv, h0)
    r1 = lambda_ * np.dot(K_inv, h1)
    r2 = np.cross(r0, r1)
    t  = lambda_ * np.dot(K_inv, h2)

    R = np.vstack((r0, r1, r2)).T
    U, S, V_t = np.linalg.svd(R)
    R = np.dot(U, V_t)

    E = np.hstack((R, t[:, np.newaxis]))

    return E

def calculate_radial_distortion(world_points, image_points, K_int, E_ext):
    u_c = np.array([K_int[0,2], K_int[1,2]])

    # Observed distortion error and Model distortion error
    d_dot = []
    D = []
    for i in range(image_points.shape[0]):
        # Projected sensor points
        P = K_int @ np.hstack([E_ext[i, :, :2], E_ext[i, :, -1].reshape((3, 1))])
        u_proj = P @ world_points.T
        u_proj /= u_proj[-1]

        d_dot.append(u_proj[:2, :].T - image_points[i])

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

def calculate_tangentical_distortion():
    raise NotImplementedError

#-------------------------------------------------------------------------------

def pack_parameters(K_int, E_ext, k_rad):
    packed_params = []

    alpha, beta, gamma, u_c, v_c = K_int[0,0], K_int[1,1], K_int[0,1], K_int[0,2], K_int[1,2]
    k0, k1 = k_rad
    packed_params.extend([alpha, beta, gamma, u_c, v_c, k0, k1])

    for E in E_ext:
        R = E[:3, :3]
        t = E[:, 3]
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

    E_ext = []
    for i in range(7, len(parameters), 6):
        rho_x, rho_y, rho_z, t_x, t_y, t_z = parameters[i:i+6]
        R = cv2.Rodrigues(np.array([rho_x, rho_y, rho_z]))[0]
        t = np.array([t_x, t_y, t_z])

        E_ext.append(np.hstack([R, t[:, np.newaxis]]))
    E_ext = np.array(E_ext)

    return K_int, E_ext, k_rad

def refine_parameters_loss_func(world_points, *params):
    K_int, E_ext, k_rad = unpack_parameters(params)

    img_points = []
    for E in E_ext:
        P = K_int @ np.hstack([E[:, :2], E[:, -1].reshape((3, 1))])
        u_proj = P @ world_points.T
        u_proj /= u_proj[-1]

        img_points.append(u_proj[:2, :].T)
    img_points = np.array(img_points)

    return img_points.flatten()

def parameter_refinement(world_points, image_points, K_int, E_ext, k_rad):

    param0 = pack_parameters(K_int, E_ext, k_rad)

    print(image_points.flatten().shape)
    popt, pcov = curve_fit(refine_parameters_loss_func, world_points, image_points.flatten(), param0)
    
    K_int, E_ext, k_rad = unpack_parameters(popt)
    return K_int, E_ext, k_rad

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    SQUARE_SIZE = 12.5
    FOLDER_NAME = 'Calibration_Images_Zhang'
    SAVE_FOLDER = 'Save_Folder'
    HEIGHT, WIDTH = (7 - 1, 10 - 1) # -1 due to only taking the inner checkerboard

    # Extract the images
    prompt = f"Loading images from {FOLDER_NAME}"
    printStart(prompt)
    images = loadImages(FOLDER_NAME)
    if DEBUG: images = images[:1]
    printEnd(prompt)

    prompt = "Extracting Image and World Points"
    printStart(prompt)
    image_corners = getImagesPoints(images, HEIGHT, WIDTH)
    world_corners = getWorldPoints(SQUARE_SIZE, HEIGHT, WIDTH)
    assert(image_corners.shape[1] == world_corners.shape[0])
    if DEBUG: displayCorners(images, image_corners, HEIGHT, WIDTH, SAVE_FOLDER)
    printEnd(prompt)

    prompt = "Estimating the Homography Matricies"
    printStart(prompt)
    H_mats = estimate_H_matrix(world_corners, image_corners)
    printEnd(prompt)

    prompt = "Estimating the B Matricies"
    printStart(prompt)
    B_mat = estimate_B_matrix(H_mats)
    printEnd(prompt)
    
    prompt = "Estimating the Internal Matrix"
    printStart(prompt)
    K_int = extract_K_int(B_mat)
    printEnd(prompt)
    print("Intrinsic Parameters:")
    print(K_int)

    prompt = "Estimating the External Matrix"
    printStart(prompt)
    E_ext = np.array([
        extract_R_t_ext(H_mat, K_int)
        for H_mat in H_mats
    ])
    printEnd(prompt)

    prompt = "Estimating the Radial Distortion"
    printStart(prompt)
    k_rad = calculate_radial_distortion(world_corners, image_corners, K_int, E_ext)
    printEnd(prompt)
    print("Radial Distortion:")
    print(f"\tk1 - {k_rad[0]}")
    print(f"\tk2 - {k_rad[1]}")

    prompt = "Preforming non-linear optimization"
    printStart(prompt)
    K_int, E_ext, k_rad = parameter_refinement(world_corners, image_corners, K_int, E_ext, k_rad)
    printEnd(prompt)
    print("Intrinsic Parameters:")
    print(K_int)
    print("Radial Distortion:")
    print(f"\tk1 - {k_rad[0]}")
    print(f"\tk2 - {k_rad[1]}")



