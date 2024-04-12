# Camera Calibration

This is an implementation of Zhang's and Tsai's camera calibration.\
\
If you need to have multiple cameras in the same coordinate frame you can either:
- Use Tsai's or Zhang's for the same object. For Zhang's this would be selecting an image as the reference coordinate frame
- Use Zhangs's for the intrinsic parameters and distortion coefficients and then use Tsai's for the extrinsics. This is nice if you are trying to calibrate other senors as well. 
\
A note to remeber is that Tsai's camera calibration assumes that $\mathbf{u_0} = \begin{bmatrix} u_0 & v_0 \end{bmatrix}^T$ is close to zero. If Zhang's camera calibration is performed then $\mathbf{u_0}$ can be used instead
