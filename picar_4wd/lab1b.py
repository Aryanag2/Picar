#!/usr/bin/env python3
import picar_4wd as fc
import time
import heapq
import argparse
import cv2
import numpy as np
import threading
from tflite_runtime.interpreter import Interpreter
# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
GRID_SIZE = 20
CELL_SIZE = 30  # 30 cm per cell
DEFAULT_START = (0, 0)
INITIAL_ORIENTATION = 1  # 0: North, 1: East, 2: South, 3: West
ORIENTATIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
DEFAULT_GOAL = (5, 0)  # 5 cells east from start
DETECTION_THRESHOLD = 0.5
MODEL_PATH = "efficientdet_lite0.tflite"
STOP_SIGN_COOLDOWN = 10.0  # Cooldown period in seconds
# Movement calibration
MOVE_FORWARD_TIME = 1.5
TURN_90_TIME = 1
OBSTACLE_PAUSE_TIME = 2.0  # Pause time after obstacle detection
# Ultrasonic sensor settings
ULTRASONIC_DISTANCE_THRESHOLD = 25  # Increased from 15 to 25 cm for better sensitivity
ULTRASONIC_CHECK_INTERVAL = 0.1  # Check more frequently (10 times per second)
ULTRASONIC_READINGS_COUNT = 3  # Take multiple readings to confirm obstacle
ULTRASONIC_READINGS_THRESHOLD = 2  # Number of readings below threshold to confirm obstacle
# Shared state with locks
camera_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
ultrasonic_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
car_pos = DEFAULT_START
goal_pos = DEFAULT_GOAL
car_orientation = INITIAL_ORIENTATION
detections = []
latest_frame = None
current_fps = 0.0
stop_sign_triggered = False
stop_sign_last_triggered = 0.0
grid_lock = threading.Lock()
frame_lock = threading.Lock()
fps_lock = threading.Lock()
position_lock = threading.Lock()
stop_sign_lock = threading.Lock()
# ==============================================================================
# PATH PLANNING (A*)
# ==============================================================================
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
def astar():
    with grid_lock:
        combined_grid = np.logical_or((camera_grid == 1), (ultrasonic_grid == 1)).astype(np.uint8)
    
    frontier = []
    heapq.heappush(frontier, (0, car_pos))
    came_from = {}
    cost_so_far = {}
    came_from[car_pos] = None
    cost_so_far[car_pos] = 0
    
    while frontier:
        current = heapq.heappop(frontier)[1]
        
        if current == goal_pos:
            break
            
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            next_node = (current[0]+dx, current[1]+dy)
            if (0 <= next_node[0] < GRID_SIZE and 
                0 <= next_node[1] < GRID_SIZE and
                combined_grid[next_node[1]][next_node[0]] == 0):
                
                new_cost = cost_so_far[current] + 1
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + heuristic(goal_pos, next_node)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current
                    
    path = []
    current = goal_pos
    while current != car_pos:
        path.append(current)
        current = came_from.get(current)
        if current is None:
            return []
    path.reverse()
    return path
# ==============================================================================
# MOVEMENT PRIMITIVES
# ==============================================================================
def move_forward():
    fc.forward(30)
    time.sleep(MOVE_FORWARD_TIME)
    fc.stop()
    time.sleep(0.2)
def turn_right():
    fc.turn_right(60)
    time.sleep(TURN_90_TIME)
    fc.stop()
    time.sleep(0.2)
def turn_left():
    fc.turn_left(60)
    time.sleep(TURN_90_TIME)
    fc.stop()
    time.sleep(0.2)
# ==============================================================================
# ULTRASONIC SENSOR
# ==============================================================================
def check_obstacle(ultrasonic):
    """Take multiple readings to confirm obstacle presence"""
    readings = []
    for _ in range(ULTRASONIC_READINGS_COUNT):
        distance = ultrasonic.get_distance()
        if 0 < distance < ULTRASONIC_DISTANCE_THRESHOLD:
            readings.append(True)
        else:
            readings.append(False)
        time.sleep(0.05)  # Short delay between readings
    
    # Return True if enough readings detected an obstacle
    return readings.count(True) >= ULTRASONIC_READINGS_THRESHOLD
# ==============================================================================
# CAMERA & DETECTION
# ==============================================================================
def detection_loop(args):
    global latest_frame, detections, camera_grid, current_fps, stop_sign_triggered, stop_sign_last_triggered
    
    pipeline = (
        'libcamerasrc ! '
        'video/x-raw,width=640,height=480,format=NV12,framerate=3/1 ! '
        'videoconvert ! video/x-raw,format=BGR ! '
        'appsink drop=1'
    )
    
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("Failed to open camera!")
        return
    try:
        interpreter = Interpreter(MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except Exception as e:
        print(f"Model load error: {e}")
        return
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                continue
            
            start_time = time.time()
            
            with frame_lock:
                latest_frame = frame.copy()
            
            input_shape = input_details[0]['shape']
            model_height, model_width = input_shape[1], input_shape[2]
            resized = cv2.resize(frame, (model_width, model_height))
            
            if input_details[0]['dtype'] == np.uint8:
                input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
            else:
                input_data = np.expand_dims(resized.astype(np.float32), axis=0)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            boxes = interpreter.get_tensor(output_details[0]['index'])[0]
            classes = interpreter.get_tensor(output_details[1]['index'])[0]
            scores = interpreter.get_tensor(output_details[2]['index'])[0]
            
            with grid_lock, position_lock:
                current_pos = car_pos
                current_orientation = car_orientation
                detections = [
                    {'id': int(c), 'score': float(s), 'box': b.tolist()}
                    for b, c, s in zip(boxes, classes, scores)
                    if s > DETECTION_THRESHOLD
                ]
                
                for det in detections:
                    dx, dy = ORIENTATIONS[current_orientation]
                    front_x = current_pos[0] + dx
                    front_y = current_pos[1] + dy
                    
                    if det['id'] == 12:  # Stop sign detection
                        # Update grid visualization
                        if 0 <= front_x < GRID_SIZE and 0 <= front_y < GRID_SIZE:
                            camera_grid[front_y][front_x] = 2
                            ultrasonic_grid[front_y][front_x] = 0
                        
                        # Handle stop sign trigger with cooldown
                        with stop_sign_lock:
                            current_time = time.time()
                            if current_time - stop_sign_last_triggered >= STOP_SIGN_COOLDOWN:
                                stop_sign_triggered = True
                                stop_sign_last_triggered = current_time
                                print(f"Stop sign detected! Cooldown started at {current_time}")
                    
                    elif det['id'] == 86:  # Traffic light detection
                        if 0 <= front_x < GRID_SIZE and 0 <= front_y < GRID_SIZE:
                            camera_grid[front_y][front_x] = 1
            
            processing_time = time.time() - start_time
            sleep_time = max(0.33 - processing_time, 0)
            time.sleep(sleep_time)
            with fps_lock:
                current_fps = 1.0 / (time.time() - start_time) if processing_time > 0 else 0.0
            
        except Exception as e:
            print(f"Detection error: {e}")
            time.sleep(1)
# ==============================================================================
# VISUALIZATION
# ==============================================================================
def create_grid_image():
    img = np.zeros((GRID_SIZE*CELL_SIZE, GRID_SIZE*CELL_SIZE, 3), dtype=np.uint8)
    img[:,:] = (0, 180, 0)
    
    with grid_lock:
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if ultrasonic_grid[y][x]:
                    cv2.rectangle(img, (x*CELL_SIZE, y*CELL_SIZE),
                                ((x+1)*CELL_SIZE, (y+1)*CELL_SIZE), (0, 0, 255), -1)
                cam_val = camera_grid[y][x]
                if cam_val == 1:
                    cv2.rectangle(img, (x*CELL_SIZE, y*CELL_SIZE),
                                ((x+1)*CELL_SIZE, (y+1)*CELL_SIZE), (0, 255, 255), -1)
                elif cam_val == 2:
                    cv2.rectangle(img, (x*CELL_SIZE, y*CELL_SIZE),
                                ((x+1)*CELL_SIZE, (y+1)*CELL_SIZE), (0, 0, 255), 2)
    path = astar()
    if path:
        pts = []
        for (x,y) in path:
            pts.append((x*CELL_SIZE + CELL_SIZE//2, y*CELL_SIZE + CELL_SIZE//2))
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts)], False, (0,255,255), 2)
    
    car_center = (car_pos[0]*CELL_SIZE + CELL_SIZE//2,
                 car_pos[1]*CELL_SIZE + CELL_SIZE//2)
    cv2.drawMarker(img, car_center, (255,0,0), 
                  markerType=cv2.MARKER_TRIANGLE_UP,
                  markerSize=CELL_SIZE, thickness=2)
    
    cv2.circle(img, (goal_pos[0]*CELL_SIZE + CELL_SIZE//2, 
              goal_pos[1]*CELL_SIZE + CELL_SIZE//2),
              CELL_SIZE//3, (255,0,255), -1)
    
    return img
def visualization_loop():
    cv2.namedWindow('AutoCar Live Map', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('AutoCar Live Map', 800, 800)
    
    while True:
        try:
            composite = np.zeros((800, 800, 3), dtype=np.uint8)
            
            with frame_lock:
                if latest_frame is not None:
                    camera_view = cv2.resize(latest_frame, (400, 300))
                    composite[:300, :400] = camera_view
            
            map_view = create_grid_image()
            map_view = cv2.resize(map_view, (600, 600))
            composite[50:650, 100:700] = map_view
            
            with fps_lock:
                fps = current_fps
            with stop_sign_lock:
                cooldown = max(0, STOP_SIGN_COOLDOWN - (time.time() - stop_sign_last_triggered))
            
            cv2.putText(composite, f"FPS: {fps:.1f}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(composite, f"Position: {car_pos}", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(composite, f"Stop Cooldown: {cooldown:.1f}s", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(composite, f"US Threshold: {ULTRASONIC_DISTANCE_THRESHOLD}cm", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            cv2.imshow('AutoCar Live Map', composite)
            if cv2.waitKey(1) == 27:
                break
            
            time.sleep(0.033)
            
        except Exception as e:
            print(f"Visualization error: {e}")
            time.sleep(1)
# ==============================================================================
# MAIN CONTROL LOGIC
# ==============================================================================
def main(args):
    global car_pos, goal_pos, car_orientation, stop_sign_triggered
    
    threading.Thread(target=detection_loop, args=(args,), daemon=True).start()
    threading.Thread(target=visualization_loop, daemon=True).start()
    
    ultrasonic = fc.Ultrasonic(fc.Pin('D8'), fc.Pin('D9'))
    fc.servo.set_angle(0)
    try:
        while True:
            # Check for obstacle using the new function that takes multiple readings
            if check_obstacle(ultrasonic):
                # Stop immediately when obstacle detected
                fc.stop()
                print("\nOBSTACLE DETECTED! Stopping and recalculating route.")
                
                with position_lock:
                    current_orient = car_orientation
                    current_pos = car_pos
                
                dx, dy = ORIENTATIONS[current_orient]
                front_x = current_pos[0] + dx
                front_y = current_pos[1] + dy
                
                with grid_lock:
                    if 0 <= front_x < GRID_SIZE and 0 <= front_y < GRID_SIZE:
                        ultrasonic_grid[front_y][front_x] = 1
                
                # Add a pause after detection
                print(f"Pausing for {OBSTACLE_PAUSE_TIME} seconds to assess the situation...")
                time.sleep(OBSTACLE_PAUSE_TIME)
                
                # Turn right to start finding a new path
                print("Turning to find new path...")
                turn_right()
                
                with position_lock:
                    car_orientation = (car_orientation + 1) % 4
                
                # Recalculate path before continuing
                print("Recalculating path...")
                continue
                
            # Check for stop sign trigger
            with stop_sign_lock:
                if stop_sign_triggered:
                    print("\nSTOP SIGN DETECTED! Waiting 3 seconds...")
                    fc.stop()
                    time.sleep(3)
                    stop_sign_triggered = False
                    
            # Path following
            path = astar()
            if not path:
                print("No valid path!")
                time.sleep(1)
                continue
                
            next_pos = path[0]
            target_dir = (next_pos[0] - car_pos[0], next_pos[1] - car_pos[1])
            
            current_dir = ORIENTATIONS[car_orientation]
            if target_dir != current_dir:
                needed_orientation = ORIENTATIONS.index(target_dir)
                turn_diff = (needed_orientation - car_orientation) % 4
                
                if turn_diff == 1:
                    turn_right()
                elif turn_diff == 3:
                    turn_left()
                elif turn_diff == 2:
                    turn_right()
                    turn_right()
                
                with position_lock:
                    car_orientation = needed_orientation
                    
            move_forward()
            
            with position_lock:
                car_pos = next_pos
                
            if car_pos == goal_pos:
                print("\n*** GOAL REACHED! ***")
                break
                
            # Short sleep to allow for frequent ultrasonic checks
            time.sleep(ULTRASONIC_CHECK_INTERVAL)
                
    except KeyboardInterrupt:
        fc.stop()
        cv2.destroyAllWindows()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto Car Controller")
    parser.add_argument('--start', type=str, default=f"{DEFAULT_START[0]},{DEFAULT_START[1]}")
    parser.add_argument('--goal', type=str, default=f"{DEFAULT_GOAL[0]},{DEFAULT_GOAL[1]}")
    parser.add_argument('--us-threshold', type=int, default=ULTRASONIC_DISTANCE_THRESHOLD,
                      help=f"Ultrasonic detection threshold in cm (default: {ULTRASONIC_DISTANCE_THRESHOLD})")
    args = parser.parse_args()
    
    car_pos = tuple(map(int, args.start.split(',')))
    goal_pos = tuple(map(int, args.goal.split(',')))
    ULTRASONIC_DISTANCE_THRESHOLD = args.us_threshold
    
    main(args)
