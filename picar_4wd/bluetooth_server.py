from bluedot.btcomm import BluetoothServer
import json
import time
import threading
import picar_4wd as fc
import signal

car_state = {
    "direction": "stopped",
    "speed": 0,
    "battery": 0,
    "temperature": 0,
    "obstacle_distance": 0,
    "timestamp": time.time()
}

state_lock = threading.Lock()

def move_forward():
    print("Moving forward")
    fc.forward(30)
    with state_lock:
        car_state["direction"] = "forward"
        car_state["speed"] = 30
    time.sleep(1) 
    stop_car()
    
def move_backward():
    print("Moving backward")
    fc.backward(30)
    with state_lock:
        car_state["direction"] = "backward"
        car_state["speed"] = 30
    time.sleep(1)  
    stop_car()

def turn_left():
    print("Turning left")
    fc.turn_left(60)
    with state_lock:
        car_state["direction"] = "left"
    time.sleep(1)  
    stop_car()

def turn_right():
    print("Turning right")
    fc.turn_right(60)
    with state_lock:
        car_state["direction"] = "right"
    time.sleep(1)  
    stop_car()

def stop_car():
    print("Stopping")
    fc.stop()
    with state_lock:
        car_state["direction"] = "stopped"
        car_state["speed"] = 0

def update_car_state():
    ultrasonic = fc.Ultrasonic(fc.Pin('D8'), fc.Pin('D9'))
    
    while True:
        try:
            temp = fc.utils.cpu_temperature()
            
            battery = fc.utils.power_read()
            
            obstacle_dist = ultrasonic.get_distance()
            if obstacle_dist < 0:
                obstacle_dist = 100  
            
            with state_lock:
                car_state["battery"] = battery
                car_state["temperature"] = temp
                car_state["obstacle_distance"] = obstacle_dist
                car_state["timestamp"] = time.time()
            
            print(f"State: Temp={temp}°C, Batt={battery}V, Dist={obstacle_dist}cm")
            
            time.sleep(0.5) 
        except Exception as e:
            print(f"Error updating state: {e}")
            time.sleep(1)

def process_command(data):
    command = data.lower().strip()
    print(f"Received command: {command}")
    
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
    
    with state_lock:
        return json.dumps(car_state)

def received_handler(data):
    print(f"Received: {data}")
    response = process_command(data)
    bluetooth_server.send(response)

update_thread = threading.Thread(target=update_car_state, daemon=True)
update_thread.start()

bluetooth_server = BluetoothServer(received_handler)
print("Bluetooth server started. Waiting for connections...")

try:
    signal.pause()
except KeyboardInterrupt:
    print("Stopping Bluetooth server...")
    fc.stop()  
    bluetooth_server.close()
