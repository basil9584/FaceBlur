How It Works

1. Face Model (Tensor-flow model for face detection) (.pb file): This is a pre-trained TensorFlow model used for face detection. The .pb file contains the model's architecture and weights.
2. Detector API (Processing detecting faces in video) (Python file): This script loads the face detection model and processes video frames to detect faces.
3. Auto-Blur Video (Blur Effect) (Python file): This script takes the output from the detector API (coordinates of detected faces) and applies a blurring effect to those areas in the video.

threshold controls the sensitivity of the detection (Ideal is 0.15)

Workflow

1. Load the Model: The detector API script loads the .pb file using TensorFlow's tf.Graph and tf.Session.
2. Process Video Frames: The video is processed frame by frame. Each frame is fed into the face detection model to get the coordinates of detected faces.
3. Blur Detected Faces: The auto-blur video script takes these coordinates and applies a blurring filter to the specified regions in each frame.
4. Save/Stream the Blurred Video: The final video with blurred faces is saved or streamed as needed.

Integration as a Backend Tool
To integrate this functionality into a server, you can follow these steps:

1. Set Up the Server Environment: Ensure your server has Python installed along with necessary libraries like TensorFlow, OpenCV, and Flask (if using a web server).
2. API Endpoint for Video Upload: Create an API endpoint to accept video files. You can use Flask to handle HTTP requests.
3. Process the Video: Upon receiving a video file, the server will process it using the face detection and blurring scripts.
4. Return/Store the Blurred Video: After processing, return the blurred video to the user or store it on the server.

Frameworks used: 
1. NumPy 
2. TensorFlow
3. OpenCV