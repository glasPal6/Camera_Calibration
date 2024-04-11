# Camera Calibration

This is an implementation of Zhang's and Tsai's camera calibration.\
\
If you need to have multiple cameras in the same coordinate frame you can either:
- Use Tsai's or Zhang's for the same object. For Zhang's this would be selecting an image as the reference coordinate frame
- Use Zhangs's for the intrinsic parameters and distortion coefficients and then use Tsai's for the extrinsics. This is nice if you are trying to calibrate other senors as well. 
