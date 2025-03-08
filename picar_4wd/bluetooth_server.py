from bluedot.btcomm import BluetoothServer
from signal import pause
def received_handler(data):
    print(data)
    s.send(data)

s = BluetoothServer(received_handler)
pause()