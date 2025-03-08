import argparse
import time
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

def draw_detections(image, boxes, classes, scores, threshold):
    """
    Draws bounding boxes and labels on the image based on detection outputs.
    
    Args:
      image: The original BGR image as a NumPy array.
      boxes: An array of bounding boxes in normalized coordinates [ymin, xmin, ymax, xmax].
      classes: An array of class IDs.
      scores: An array of detection scores.
      threshold: Minimum confidence score required to draw a box.
      
    Returns:
      The annotated image.
    """
    height, width, _ = image.shape
    for i in range(len(scores)):
        if scores[i] < threshold:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        left = int(xmin * width)
        top = int(ymin * height)
        right = int(xmax * width)
        bottom = int(ymax * height)
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"ID {int(classes[i])}: {scores[i]:.2f}"
        cv2.putText(image, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image

def is_target_close(target_id, boxes, classes, scores, score_threshold=0.25, area_threshold=0.25):
    """
    Checks whether any detection for the given target_id is considered "close".
    
    For each detection with a confidence score above score_threshold, the normalized
    bounding box area is calculated (assuming the values in the box are in [0, 1]).
    If the box area is above area_threshold, that detection is considered close.
    
    Args:
      target_id: The integer ID to check (e.g., 12 for a stop sign).
      boxes: An array of bounding boxes, each in [ymin, xmin, ymax, xmax].
      classes: An array of class IDs.
      scores: An array of detection scores.
      score_threshold: Minimum confidence score required.
      area_threshold: Minimum normalized bounding box area required to be considered close.
      
    Returns:
      A tuple (True, detection_info) if the target is seen close by; otherwise (False, None).
      detection_info is a dictionary containing the id, score, box, and area.
    """
    for i in range(len(scores)):
        if scores[i] < score_threshold:
            continue
        if int(classes[i]) == target_id:
            ymin, xmin, ymax, xmax = boxes[i]
            area = (ymax - ymin) * (xmax - xmin)
            if area >= area_threshold:
                return True, {"id": int(classes[i]), "score": scores[i], "box": boxes[i], "area": area}
    return False, None

def get_biggest_detection(boxes, classes, scores, score_threshold=0.25):
    """
    Iterates over all detections with confidence above score_threshold and returns
    the detection with the largest normalized bounding box area.
    
    Args:
      boxes: An array of bounding boxes [ymin, xmin, ymax, xmax] with normalized coordinates.
      classes: An array of class IDs.
      scores: An array of detection scores.
      score_threshold: Minimum score to consider a detection valid.
      
    Returns:
      A dictionary containing 'id', 'score', 'box', and 'area' for the biggest detection.
      Returns None if no detection passes score_threshold.
    """
    biggest_index = None
    biggest_area = 0.0
    for i in range(len(scores)):
        if scores[i] < score_threshold:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        area = (ymax - ymin) * (xmax - xmin)
        if area > biggest_area:
            biggest_area = area
            biggest_index = i
    if biggest_index is not None:
        return {"id": int(classes[biggest_index]),
                "score": scores[biggest_index],
                "box": boxes[biggest_index],
                "area": biggest_area}
    else:
        return None

def get_camera_pipeline(frame_width, frame_height):
    """
    Constructs the GStreamer pipeline string to capture frames from the Raspberry Pi camera.
    """
    pipeline = (
        f"libcamerasrc ! video/x-raw,width={frame_width},height={frame_height},format=NV12,framerate=30/1 ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink"
    )
    return pipeline

def initialize_camera(frame_width, frame_height):
    """
    Opens the camera using the specified GStreamer pipeline.
    """
    pipeline = get_camera_pipeline(frame_width, frame_height)
    print("Using pipeline:", pipeline)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Error: Failed to open camera using pipeline.")
        raise RuntimeError("Camera could not be opened")
    return cap

def initialize_interpreter(model):
    """
    Initializes the TFLite interpreter and allocates the required tensors.
    """
    interpreter = Interpreter(model_path=model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def preprocess_frame(frame, model_width, model_height, is_uint8_model, input_scale=None, input_mean=None):
    """
    Resizes and converts the image from BGR to RGB, then performs quantization or normalization.
    """
    resized = cv2.resize(frame, (model_width, model_height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if is_uint8_model:
        input_data = np.expand_dims(rgb.astype(np.float32), 0)
        input_data = ((input_data / 255.0) / input_scale + input_mean).astype(np.uint8)
    else:
        input_data = np.expand_dims(rgb.astype(np.float32) / 255.0, 0)
    return input_data

def run_inference(interpreter, input_details, input_data):
    """
    Sets the input tensor, invokes the interpreter, and retrieves the output.
    """
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]      # Bounding box predictions.
    classes = interpreter.get_tensor(output_details[1]['index'])[0]    # Class IDs.
    scores = interpreter.get_tensor(output_details[2]['index'])[0]     # Confidence scores.
    return boxes, classes, scores

def run(model, max_results, score_threshold, frame_width, frame_height, close_area_threshold=0.25, target_id=None):
    """
    Continuously captures frames from the camera using a GStreamer pipeline, runs inference using
    a TFLite model, and prints detection information to the terminal.
    
    In each frame, the function prints the information about the biggest detected object (if any).
    Additionally, if a target_id is provided, it checks if that object is very close (based on the
    relative area of its bounding box) and prints a message if so.
    
    Args:
      model: Path to the TFLite model file.
      max_results: (Unused here but kept for compatibility) Maximum number of detection results.
      score_threshold: Minimum confidence score to consider a detection.
      frame_width: Width of the camera frame.
      frame_height: Height of the camera frame.
      close_area_threshold: Normalized area above which a detection is considered "close."
      target_id: (Optional) Specific target ID to check for closeness.
    """
    cap = initialize_camera(frame_width, frame_height)
    interpreter, input_details, output_details = initialize_interpreter(model)
    
    # Get model input dimensions.
    input_shape = input_details[0]['shape']  # typically [1, height, width, 3]
    model_height, model_width = input_shape[1], input_shape[2]
    
    is_uint8_model = input_details[0]['dtype'] == np.uint8
    if is_uint8_model:
        input_mean = input_details[0]['quantization'][1]
        input_scale = input_details[0]['quantization'][0]
    else:
        input_mean, input_scale = None, None

    fps_counter = 0
    fps = 0.0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Preprocess frame to match the model's input requirements.
            input_data = preprocess_frame(frame, model_width, model_height,
                                          is_uint8_model, input_scale, input_mean)
            # Run inference.
            boxes, classes, scores = run_inference(interpreter, input_details, input_data)
            
            # Annotate image using detection outputs.
            annotated_frame = draw_detections(frame.copy(), boxes, classes, scores, score_threshold)
            
            # Get the biggest detection from the frame.
            biggest_detection = get_biggest_detection(boxes, classes, scores, score_threshold)
            if biggest_detection:
                print(f"Biggest Detection: ID: {biggest_detection['id']}, "
                      f"Score: {biggest_detection['score']:.2f}, "
                      f"Area: {biggest_detection['area']:.2f}")
            else:
                print("No valid detection.")
            
            # If a target_id is provided, check if that object is close.
            if target_id is not None:
                close, det_info = is_target_close(target_id, boxes, classes, scores,
                                                  score_threshold, close_area_threshold)
                if close:
                    print(f"Target object (ID {target_id}) is very close: Score {det_info['score']:.2f}, "
                          f"Area: {det_info['area']:.2f}")
            
            fps_counter += 1
            if fps_counter % 10 == 0:
                end_time = time.time()
                fps = 10 / (end_time - start_time)
                start_time = time.time()
                print(f"FPS: {fps:.2f}")
            
            # Display the annotated frame in a window.
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Object Detection", annotated_frame)

            # Press ESC key to exit.
            if cv2.waitKey(1) == 27:
                print("Exiting detection loop...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        description="TFLite Object Detection with Proximity Checking using OpenCV on Raspberry Pi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=str, default="efficientdet_lite0.tflite",
                        help="Path to the TensorFlow Lite model file.")
    parser.add_argument("--maxResults", type=int, default=5,
                        help="Maximum number of detection results (model output decides).")
    parser.add_argument("--scoreThreshold", type=float, default=0.25,
                        help="Score threshold for displaying detections.")
    parser.add_argument("--frameWidth", type=int, default=640,
                        help="Width of the camera frame.")
    parser.add_argument("--frameHeight", type=int, default=480,
                        help="Height of the camera frame.")
    # Optional target id to check for proximity (e.g., 12 for a stop sign)
    parser.add_argument("--targetId", type=int, default=None,
                        help="(Optional) Target object ID to check for proximity.")
    # Optional area threshold, in normalized units, to decide if an object is 'close'
    parser.add_argument("--closeAreaThresh", type=float, default=0.25,
                        help="Normalized area threshold to consider an object as close.")
    args = parser.parse_args()
    
    run(args.model, args.maxResults, args.scoreThreshold,
        args.frameWidth, args.frameHeight,
        close_area_threshold=args.closeAreaThresh,
        target_id=args.targetId)

if __name__ == "__main__":
    main()
