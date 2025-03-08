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
    time.sleep(1)  # Move for 1 second
    stop_car()
    
def move_backward():
    print("Moving backward")
    fc.backward(30)
    with state_lock:
        car_state["direction"] = "backward"
        car_state["speed"] = 30
    time.sleep(1)  # Move for 1 second
    stop_car()

def turn_left():
    print("Turning left")
    fc.turn_left(60)
    with state_lock:
        car_state["direction"] = "left"
    time.sleep(1)  # Turn for 1 second
    stop_car()

def turn_right():
    print("Turning right")
    fc.turn_right(60)
    with state_lock:
        car_state["direction"] = "right"
    time.sleep(1)  # Turn for 1 second
    stop_car()

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

def handle_client(client_socket, addr):
    """Handle an individual client connection"""
    print(f"Client connected: {addr}")
    try:
        while True:
            # Receive command
            data = client_socket.recv(1024)
            if not data:
                print(f"Client disconnected: {addr}")
                break
            
            command = data.decode('utf-8').strip()
            print(f"Received command from {addr}: {command}")
            
            # Process command
            if command == "forward":
                threading.Thread(target=move_forward).start()
            elif command == "backward":
                threading.Thread(target=move_backward).start()
            elif command == "left":
                threading.Thread(target=turn_left).start()
            elif command == "right":
                threading.Thread(target=turn_right).start()
            elif command == "stop":
                stop_car()
            
            # Send current state back to client
            with state_lock:
                response = json.dumps(car_state)
            client_socket.sendall(response.encode('utf-8'))
    except Exception as e:
        print(f"Error handling client {addr}: {e}")
    finally:
        client_socket.close()
        print(f"Connection with {addr} closed")

# --- TCP Server Setup ---
def start_server():
    HOST = ""   # Bind to all interfaces
    PORT = 65432

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(5)  # Allow up to 5 pending connections
        print(f"Server listening on port {PORT}...")
        
        while True:
            try:
                client, addr = s.accept()
                # Start a new thread for each client
                client_thread = threading.Thread(target=handle_client, args=(client, addr), daemon=True)
                client_thread.start()
            except Exception as e:
                print(f"Error accepting connection: {e}")
                time.sleep(1)

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