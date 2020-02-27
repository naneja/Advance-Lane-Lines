# Advance Lane Lines
> Steps
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
* Apply a distortion correction to raw images
* Use color transforms, gradients, etc., to create a thresholded binary image
* Apply a perspective transform to rectify binary image ("birds-eye view")
* Detect lane pixels and fit to find the lane boundary
* Measure Curvature


## Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
Camera introduces two types of distortation:
* Radial Distortion
    * Real cameras use curved lenses to form an image, and light rays often bend a little too much or too little at the edges of these lenses. This creates an effect that distorts the edges of images, so that lines or objects appear more or less curved than they actually are. This is called radial distortion, and it’s the most common type of distortion.
* Tangential Distortion
    * This occurs when a camera’s lens is not aligned perfectly parallel to the imaging plane, where the camera film or sensor is. This makes an image look tilted so that some objects appear farther away or closer than they actually are.
    
* Distortion can be corrected with following Coefficients and Correction
    * radial distortion: k1, k2, and k3
    * tangential distortion: p1 and p2
    
In order to calibrate camera, following 20 images were used.
![](images/calibration.png)<br>
*Calibration Images for Object Points for 9 inside corners on x-axis and 6 inside corners on y-axis with z-value zero *

Following process explains Calibration Steps

> Get Object Points  
objp = np.zeros((6*9, 3), np.float32)  
mat = np.mgrid[0:9, 0:6]  
mat = mat.T  
mat = mat.reshape(-1, 2)  
objp[:, :2] = mat   

> Image Points  
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)  
imgpoints.append(corners)  
objpoints.append(objp)  

> Calibrate to calculate distortion coefficients  
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)  
*save mtx and dist to be used later for all images  

>  Test undistortion on an image  
undist = cv2.undistort(image, mtx, dist, None, mtx)  

![](images/undistort_sample.png)  
*Sample Undistorted Image*


> Transform Perspective  
gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)  
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)  

> Get Source Points from Corners  
top_left, top_right = corners[0], corners[nx-1]  
bottom_right, bottom_left = corners[-1], corners[-nx]  
src = np.float32([top_left, top_right, bottom_right, bottom_left])  

> Get Destination Points from Image with offset e.g. 300  
top_left, top_right = [offset, offset], [image_size[0] - offset, offset]    
bottom_right = [image_size[0] - offset, image_size[1] - offset]    
bottom_left = [offset, image_size[1] - offset]  
dst = np.float32([top_left, top_right, bottom_right, bottom_left])  

> Perspective transform matrix  
M = cv2.getPerspectiveTransform(src, dst)

> Get Warped Image  
warped = cv2.warpPerspective(undist, M, image_size)  


![](images/undistort_warp_sample.png)  
*Sample Undistorted and Warped Image*


## Apply a distortion correction to raw images
![](images/undistort_sample.png)<br>
*Sample Image Undistorted*


## Use color transforms, gradients, etc., to create a thresholded binary image.
![](images/threshold_binary_images.png)<br>
*Threshold Binary Images*

## Apply a perspective transform to rectify binary image ("birds-eye view").
![](images/sample_corners.png)<br>
*Object Points Marked on Sample Image with Red Star for Bird-Eye View*

![](images/bird_eye_images.png)<br>
*Threshold Binary Images from Bird-Eye View*


## Detect lane pixels and fit to find the lane boundary
![](images/sliding_sample.png)<br>
*Lane Detection using Sliding Window*

![](images/sample_lane.png)<br>
*Sample Lane without Sliding Window*

![](images/lane_images.png)<br>
* Finding Lanes in Sample Images 


## Measure Curvature
![](images/sample_output_bird_eye.png)<br>
*Sample output image with Curvature and Centre in Bird Eye View*

![](images/sample_output.png)<br>
*Sample output image with Curvature and Centre in Camera View*


# Input and Final Output

Input: [Project Video](data/project_video.mp4)

Output: [Project Video Marked](data/project_video_marked.mp4)


