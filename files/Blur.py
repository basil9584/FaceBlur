import os
import argparse
import cv2
from Detection import Detector


def obscure_regions(image, regions):
    """
    Blurs regions of interest in an image.

    Args:
    image -- the image to be edited as a matrix
    regions -- list of regions to be blurred, each element is a dictionary with [id, score, x1, y1, x2, y2] keys

    Returns:
    image -- the blurred image as a matrix
    """
    for region in regions:
        # Extract region coordinates
        x1, y1 = region["x1"], region["y1"]
        x2, y2 = region["x2"], region["y2"]

        # Crop the region from the image
        cropped_region = image[y1:y2, x1:x2]

        # Apply Gaussian blur to the cropped region
        blurred_region = cv2.blur(cropped_region, (25, 25))

        # Replace the original region with the blurred one
        image[y1:y2, x1:x2] = blurred_region

    return image


def process_video(args):
    # Set model path and detection threshold
    model_path = args.model_path
    detection_threshold = args.threshold

    # Create a face detector object
    detector = Detector(model_path=model_path, name="detection")

    # Open the input video file
    video_capture = cv2.VideoCapture(args.input_video)

    # Get video properties
    video_width = int(video_capture.get(3))
    video_height = int(video_capture.get(4))
    video_fps = video_capture.get(5)

    # Set up output video writer if specified
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(args.output_video, fourcc, 20.0, (video_width, video_height))

    frame_count = 0
    while True:
        # Read a frame from the video
        _, frame = video_capture.read()
        frame_count += 1

        # Check if end of video is reached
        if frame is None:
            break

        # Exit on 'q' key press
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        # Perform face detection
        detected_faces = detector.detect_objects(frame, threshold=detection_threshold)

        # Blur detected faces
        blurred_frame = obscure_regions(frame, detected_faces)

        # Display the blurred frame
        cv2.imshow('Blurred Video', blurred_frame)

    # Save the output video if specified
    if args.output_video:
        output_video.write(blurred_frame)
        print('Blurred video has been saved successfully at', args.output_video, 'path')

    # Clean up
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Video Blurring Parameters')

    # Add arguments
    parser.add_argument('-i', '--input_video', help='Path to the input video', type=str, required=True)
    parser.add_argument('-m', '--model_path', help='Path to the .pb model', type=str, required=True)
    parser.add_argument('-o', '--output_video', help='Output file path', type=str)
    parser.add_argument('-t', '--threshold', help='Face detection confidence', default=0.7, type=float)
    args = parser.parse_args()

    # Validate input file path
    assert os.path.isfile(args.input_video), 'Invalid input file'

    # Validate output directory if specified
    if args.output_video:
        assert os.path.isdir(os.path.dirname(args.output_video)), 'No such directory'

    # Process the video
    process_video(args)