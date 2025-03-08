#!/usr/bin/env python3
import socket
import json
import time
import threading
import picar_4wd as fc
import os

# Global variables to track car state
car_state = {
    "direction": "stopped",
    "speed": 0,
    "battery": 0,
    "temperature": 0,
    "obstacle_distance": 0,
    "timestamp": time.time()
}

# Lock for thread-safe access to car_state
state_lock = threading.Lock()

# --- Car control functions ---
def move_forward():
    print("Moving forward")
    fc.forward(30)
    with state_lock:
        car_state["direction"] = "forward"
        car_state["speed"] = 30
    
def move_backward():
    print("Moving backward")
    fc.backward(30)
    with state_lock:
        car_state["direction"] = "backward"
        car_state["speed"] = 30

def turn_left():
    print("Turning left")
    fc.turn_left(60)
    with state_lock:
        car_state["direction"] = "left"

def turn_right():
    print("Turning right")
    fc.turn_right(60)
    with state_lock:
        car_state["direction"] = "right"

def stop_car():
    print("Stopping")
    fc.stop()
    with state_lock:
        car_state["direction"] = "stopped"
        car_state["speed"] = 0

def update_car_state():
    """Update car state with actual sensor data"""
    # Initialize the ultrasonic sensor
    ultrasonic = fc.Ultrasonic(fc.Pin('D8'), fc.Pin('D9'))
    
    while True:
        try:
            # Get CPU temperature - real data
            temp = fc.utils.cpu_temperature()
            
            # Get battery level - real data
            battery = fc.utils.power_read()
            
            # Get obstacle distance from ultrasonic sensor
            obstacle_dist = ultrasonic.get_distance()
            # Handle invalid readings
            if obstacle_dist < 0:
                obstacle_dist = 100  # Set to a default value when out of range
            
            # Update state values
            with state_lock:
                car_state["battery"] = battery
                car_state["temperature"] = temp
                car_state["obstacle_distance"] = obstacle_dist
                car_state["timestamp"] = time.time()
            
            # Print current state for debugging
            print(f"State: Temp={temp}°C, Batt={battery}V, Dist={obstacle_dist}cm")
            
            time.sleep(0.5)  # Update twice per second
        except Exception as e:
            print(f"Error updating state: {e}")
            time.sleep(1)

def process_command(command):
    """Process received command and control the car"""
    command = command.lower().strip()
    
    if command == "forward":
        move_forward()
    elif command == "backward":
        move_backward()
    elif command == "left":
        turn_left()
    elif command == "right":
        turn_right()
    elif command == "stop":
        stop_car()
    elif command == "status":
        # Just return current status, no need to change car state
        pass
    else:
        print(f"Unknown command: {command}")
    
    # Return current state regardless of command
    with state_lock:
        return dict(car_state)  # Return a copy of the state

# --- TCP Server Setup ---
def start_server():
    HOST = ""   # Bind to all interfaces
    PORT = 65432

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on port {PORT}...")
        
        while True:
            try:
                client, client_addr = s.accept()
                with client:
                    print(f"Connected by {client_addr}")
                    data = client.recv(1024)
                    if not data:
                        continue
                    
                    command = data.decode().strip()
                    print(f"Received command: {command}")
                    
                    response = process_command(command)
                    client.sendall(json.dumps(response).encode())
            except Exception as e:
                print(f"Connection error: {e}")

if __name__ == "__main__":
    try:
        # Start state update thread
        state_thread = threading.Thread(target=update_car_state, daemon=True)
        state_thread.start()
        
        # Start server
        start_server()
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        # Clean up
        fc.stop()
        print("Server shutdown, car stopped")