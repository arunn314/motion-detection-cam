# motion-detection-cam
Raspberry Pi based Security Cam to detect motion and upload images to Google Drive.

Detect motion near the camera and send alerts to FB messenger.
The script detects motion based on contours formed after background subtraction using Gaussian Blur and thresholding.

The background image is updated continuously using Exponential weighted Average of last frames to adapt to the changes in the environment.

The frames with motion are uploaded to Google Drive. The images will be stored for 1 week.