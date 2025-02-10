import serial
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.properties import StringProperty
from kivy.core.window import Window
from kivy.uix.gridlayout import GridLayout
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
import serial.tools.list_ports
from kivy.utils import platform

# Set the window size
Window.size = (1080, 2340)

class IconButton(ButtonBehavior, Image):
    pass

class CommandButton(Button):
    command = StringProperty('')

    def __init__(self, **kwargs):
        super(CommandButton, self).__init__(**kwargs)
        self.background_normal = ''
        self.background_color = (0.2, 0.6, 1, 1)  # Light blue
        self.color = (1, 1, 1, 1)  # White text
        self.font_size = 18
        self.size_hint = (0.45, 0.15)
        self.bind(on_press=self.send_command)

    def send_command(self, instance):
        app = App.get_running_app()
        if app.ser:
            try:
                app.ser.write(self.command.encode())
                print(f"Sent command: {self.command}")
            except serial.SerialException as e:
                print(f"Error sending command: {e}")

class MotorControlApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Add motor control buttons
        button_layout = GridLayout(cols=2, spacing=10, size_hint=(1, 0.8))
        button_layout.add_widget(CommandButton(text='Forward', command='F'))
        button_layout.add_widget(CommandButton(text='Backward', command='B'))
        button_layout.add_widget(CommandButton(text='Left', command='L'))
        button_layout.add_widget(CommandButton(text='Right', command='R'))
        layout.add_widget(button_layout)

        # Add stop button
        stop_button = CommandButton(text='Stop', command='S')
        stop_button.background_color = (1, 0, 0, 1)  # Red
        stop_button.size_hint = (1, 0.2)
        layout.add_widget(stop_button)

        # Add label for connection status
        self.status_label = Label(text="Connecting to Arduino...", size_hint=(1, 0.1))
        layout.add_widget(self.status_label)

        # Add check connection button
        check_button = Button(text='Check Connection', size_hint=(1, 0.1))
        check_button.bind(on_press=self.check_connection)
        layout.add_widget(check_button)

        # Attempt to connect to Arduino
        self.ser = None
        self.connect_to_arduino()

        return layout

    def connect_to_arduino(self):
        if platform == 'android':
            from usb4a import usb
            from usbserial4a import serial4a

            usb_device_list = usb.get_usb_device_list()
            if not usb_device_list:
                self.status_label.text = "No USB device found"
                print("No USB device found")
                return

            for device in usb_device_list:
                try:
                    self.ser = serial4a.get_serial_port(device.getDeviceName(), 9600, 8, 'N', 1, timeout=1)
                    if self.ser:
                        self.status_label.text = f"Connected to Arduino on {device.getDeviceName()}"
                        print(f"Connected to Arduino on {device.getDeviceName()}")
                        return
                except Exception as e:
                    print(f"Failed to connect on {device.getDeviceName()}: {e}")
            self.status_label.text = "Failed to connect to any Arduino device"
            print("Failed to connect to any Arduino device")
        else:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                try:
                    self.ser = serial.Serial(port.device, 9600, timeout=1)
                    self.status_label.text = f"Connected to Arduino on {port.device}"
                    print(f"Connected to Arduino on {port.device}")
                    return
                except serial.SerialException as e:
                    print(f"Failed to connect on {port.device}: {e}")
            self.status_label.text = "Failed to connect to any Arduino device"
            print("Failed to connect to any Arduino device")

    def check_connection(self, instance):
        if self.ser and self.ser.is_open:
            self.status_label.text = "Arduino is connected."
            print("Arduino is connected.")
        else:
            self.status_label.text = "Arduino is not connected."
            print("Arduino is not connected.")

    def on_stop(self):
        # Clean up serial connection when app closes
        if self.ser:
            self.ser.close()
            print("Serial connection closed.")

if __name__ == "__main__":
    MotorControlApp().run()
