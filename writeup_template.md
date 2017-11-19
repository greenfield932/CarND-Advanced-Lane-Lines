# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/original.jpg
[image2]: ./examples/undistorted.jpg
[image3]: ./examples/Color_pipe_line_0.jpg
[image4]: ./examples/Color_pipe_line_1.jpg
[image5]: ./examples/Color_pipe_line_2.jpg
[image6]: ./examples/ROI.jpg
[image7]: ./examples/Bird_eye_view1.jpg
[image8]: ./examples/histogram.png
[image9]: ./examples/Sliding_windows.jpg
[image10]: ./examples/Fitting_curves.jpg
[image11]: ./examples/Output.jpg
[image12]: ./examples/output_project_video.avi 

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/greenfield932/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a writeup for this project.

Link provided above.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines #20 through #56 of the file called `camera.py`).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objectPoints` is just a replicated array of coordinates, and `pattern_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imagePoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objectPoints` and `imagePoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image3]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image1]

The resulted undistorted image provided below:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #89 through #124 in `utilities.py`).  Here's an example of output for this step (red color data obtained from color treshold and green color - data obtained from gradient tresholds).

![alt text][image4]

Resulted binary mask looks like this:

![alt text][image5]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines #127 through #133 in the file `utilities.py` (./utilities.py). The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
    src  = np.float32([
        [left_bottom[0], left_bottom[1]],
        [left_top[0], left_top[1]],
        [right_top[0], right_top[1]],
        [right_bottom[0], right_bottom[1]]
    ])

    dst  = np.float32([
        [0, ysize-1],
        [0,0],
        [xsize-1, 0],
        [xsize-1, ysize-1]
    ])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 130, 670      | 0, 719        | 
| 550, 470      | 0, 0          |
| 760, 470      | 1279, 0       |
| 1180, 670     | 1279, 719     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

The code for this procedure is located at the function `slidingWindowsFindRawPixelsIndexes()` at lines #9 - #83 and function `fitCurves()` #94 - #140 of `sliding_windows.py`.

The code mostly kept as is provided by Udacity. The algorithm is based one the following steps:

##### 1) Find peaks on the bottom of the image to detect start of lane lines

![alt text][image8]

##### 2) Based on peaks start search for pixels by sliding windows stacked from bottom to top along the white pixels

![alt text][image9]

##### 3) Fit polynomial on the detected points to represent line as 3-coefficient parameters array for equation.

![alt text][image10]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #229 - #249 in my code in `sliding_windows.py` using the equations and reference code provided by Udacity. The main idea is based on mathematical equations and conversion from pixel coordinates to world (meters) coordinate using predefined translation coefficient.

Position of the vehicle is calculated at #38-#46 `path.py` based on an assumption that vehicle center fit middle point of the lines when car is located at the center. To find a shift I used start points of the found lines to calculate a middle point coordinate X, then substracted from X center of the image (width/2) and resulted value multiplied by translation coefficient from pixels to meters.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #251 through #270 in my code in `sliding_windows.py` in the function `drawTargetLane()`.  Here is an example of my result on a test image:

![alt text][image11]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./examples/output_project_video.avi)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline is quite stable on project video, but fails on challenge and hard_challenge videos.

One of the problems I faced were false positives on stright lines on the road (a joint of a road coverages) appear in gradient filter very actively, as result some of them can be interpreted as lane line. This problem can be fixed by using other color treshold methods based on white/yellow color extraction instead of gradient filter, or their combination in 'AND' manner (instead of 'OR
). 

Next problem on hard_challenge video appears due to inability to use fixed region of interest. I believe adaptive region of interest adjusted by curvature calculated from lines can be used here.

Also I faced a problem with very sunny places where impossible to determine road lines, in addition turns on the road at the same time makes averaging algorithm fail, since it should change very fast according to the road turns, but at the same time it can't get new frames due to sun shines.
I believe preprocessing the same color image several times with different treshold with feedback from analysis algorithm can make pipeline more robust.