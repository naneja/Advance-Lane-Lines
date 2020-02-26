# Advance Lane Lines
> Steps
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
* Apply a distortion correction to raw images
* Use color transforms, gradients, etc., to create a thresholded binary image
* Apply a perspective transform to rectify binary image ("birds-eye view")
* Detect lane pixels and fit to find the lane boundary
* Measure Curvature




## Compute the camera calibration matrix and distortion coefficients given a set of chessboard images

![](images/calibration.png)<br>
*Calibration Images*


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
*Sample Lane with Sliding Window*

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


