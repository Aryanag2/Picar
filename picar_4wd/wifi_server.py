#!/usr/bin/env python3
import socket
import json
import time
import picar_4wd as fc  # Make sure this library is installed and properly configured

# --- Car control functions ---
def move_forward():
    print("Moving forward")
    fc.forward(30)  # Set speed value as needed
    time.sleep(1.5)  # Adjust time as needed
    fc.stop()

def turn_left():
    print("Turning left")
    fc.turn_left(60)
    time.sleep(1)
    fc.stop()

def turn_right():
    print("Turning right")
    fc.turn_right(60)
    time.sleep(1)
    fc.stop()

def get_sensor_info():
    # Replace these with actual sensor function calls if available.
    # Here we simulate sensor values.
    battery = 80  # Example: fc.get_battery_level() if available
    speed = 30    # For instance, measured speed
    distance = 150  # Example: total distance traveled
    return {"battery": battery, "speed": speed, "distance": distance}

def process_command(command):
    command = command.lower()
    if command == "forward":
        move_forward()
    elif command == "left":
        turn_left()
    elif command == "right":
        turn_right()
    else:
        print("Unknown command received:", command)
    # After executing the command, get sensor info
    return get_sensor_info()

# --- TCP Server Setup ---
HOST = ""   # Bind to all interfaces (or specify your Pi's IP)
PORT = 65432

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Server listening on port {PORT}...")
    while True:
        client, client_addr = s.accept()
        with client:
            print("Connected by", client_addr)
            data = client.recv(1024)  # Read up to 1024 bytes
            if not data:
                continue
            command = data.decode().strip()
            print("Received command:", command)
            response = process_command(command)
            # Send JSON-encoded response back to the client
            client.sendall(json.dumps(response).encode())
